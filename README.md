# Triton 算子优化研究

> 本仓库是脱敏源码快照。原始 Git 历史、GPU 实验包和私人 notebook 未上传。历史实测对应本地冻结版本；当前源码可用于重新运行新实验。参见[数据可用性说明](docs/data-availability.md)。

这是独立实验仓库，研究 GEMM、融合 Attention、受限 Conv2D 的分块、访存和融合优化。**必做技术实验已通过完整验收，被测固定版本为 `54f29c2`。** 两个独立 L4 会话各完成 30 个任务、1,284 条算子及模块记录；开发调优、逐因素消融、冻结策略、冷/热编译缓存成本、安全范围及硬件产物均已核验。A100 补充会话也已单独验收。技术完成不包括论文、研究报告或答辩材料，也不表示所有算子都优于框架。执行与审计方法见[技术实验执行约定](docs/study-workflow.md)。

## 技术验收与证据

| 项目 | 已核验覆盖 |
|---|---|
| 技术实验阶段 Mac 验收 | 23 项门禁；148 项测试通过，96 项 GPU 测试明确跳过 |
| 整理与分享检查 | 新增 8 项隐私保护测试；当前本地全量 156 项通过、96 项 GPU 测试跳过 |
| L4 数值验证 | 28 项 CUDA 检查、96 项 GPU 正确性测试 |
| 开发与消融 | 20 个任务、1,488 条记录，13 个 GEMM 逐因素配置、384 组消融比较 |
| 冻结方法 | 按精度分别拟合形状规则；最近三个开发邻居，短名单最多三项；独立集不参与拟合 |
| 独立评测和复验 | 每会话 24 个案例、双协议各三轮；完整 Transformer/CNN 前向包含在内 |
| A100 补充 | 单会话 28 项 CUDA 检查、96 项 GPU 测试、30 个任务和 1,284 条记录；策略直接沿用 L4 冻结版本 |
| 编译缓存成本 | 84 个任务、42 对受控冷/热缓存，逐轮启动/搜索成本及摊销公式 |
| 硬件与安全 | 16 组 IR/二进制/SASS、6 个 Nsight launch；限定项目安全范围见下文 |

完整性审计记录在工作区 [completion-audit.json](docs/data-availability.md)，同目录保存各阶段逐案例比较、跨会话一致性、消融、硬件交叉检查和成本 JSON。原始 ZIP、CSV、轨迹与命令日志保留在 `storage/runs/triton/`；[要求清单](docs/data-availability.md)关联全部证据。该次审计使用分析版本 `53789d2`，核验 GPU 工作负载、配置与锁文件仍与被测版本一致。

两个 L4 会话分别有六条 Attention 记录缺少内核名称，已用相同环境、种子和已选配置补采不计时的 GPU 轨迹。原始空记录、时延、配置及全部补采尝试均保留。一次被 Colab 回收、未能回收原始证据的早期复验尝试不计入验收；后续独立会话已完整归档。

项目安全检查有明确边界：完整应用的原始 initcheck/racecheck 失败仍保留。归属补充使用精确的厂商报告例外和故障检出对照，保留全部项目内核的初始化插桩；没有认证厂商共享内存初始化安全，也没有宣称原告警是误报。GPU 检查的详细范围由完成审计中的 `safety_scope` 记录。

性能 JSON 保留所有退化和方向变化，`cuda_event` 与 `cuda_graph` 分开。Graph 要求静态地址契约；普通调用包含主机提交间隙，两者不能互换为统一加速倍数。Transformer 相对 `compile_mm_compat` 的退化、部分算子相对成熟库的劣势均未删除。微小或跨轮正负不一致的成本收益不支持稳定回本承诺。

A100 的[单独审计](docs/data-availability.md)及[逐案例比较](docs/data-availability.md)与 L4 证据分开保存；没有在 A100 上重新拟合策略，也不合并硬件计算整体加速率。该补充只有一个完整会话，没有重复 L4 的安全资格与硬件计数器采集，不代表 A100 已获相同安全认证。

A100 首次导出未能取回原始包，其成功文字及哈希不计入验收。补跑的原始 ZIP 已同时通过普通下载和 notebook 内完整字节备份回收，两份数据逐字节一致；[恢复记录](docs/data-availability.md)关联带输出的笔记本、字节数与 SHA256。此前的未验收尝试及恢复原因仍保留。

账号、订阅和算力余额记录仅在本地保留。

## 实测示例的使用范围

以下均为 FP16，分别给出两个独立 L4 会话各三轮的配对加速比中位数。它们是便于阅读的具体示例；[全部 1,356 组跨会话对照](docs/data-availability.md)保留所有形状、精度、后端、退化及方向变化。

| 案例 | 计时协议 | 基线 | 两会话加速比 | 两会话延迟减少 |
|---|---|---|---|---|
| 预绑定 GEMM+Bias+GELU；M/N/K=129/193/257 | `cuda_event` | `torch` | 2.026× / 2.013× | 50.64% / 50.33% |
| Attention；B/H/N/D=1/2/65/32，非 causal | `cuda_graph` | `sdpa_flash` | 1.636× / 1.640× | 38.89% / 39.01% |
| 同一 Attention 的普通调用 | `cuda_event` | `sdpa_flash` | 0.858× / 0.869× | -16.61% / -15.12% |
| Conv2D；N/C/H/W/O/K=1/16/32/32/32/3，stride=1、padding=1 | `cuda_graph` | `torch` | 3.144× / 3.142× | 68.19% / 68.17% |
| TinyCNN；输入 1/3/31/29 | `cuda_graph` | `compile` | 1.453× / 1.474× | 31.18% / 32.14% |
| TinyTransformer；B/N/D=1/127/64 | `cuda_graph` | `compile_mm_compat` | 0.908× / 0.910× | -10.12% / -9.83% |

加速比为基线耗时除以 Triton 耗时；延迟减少为 `100 × (1 − 1/加速比)`，负值表示更慢。预绑定 GEMM 使用 `triton_prepared_rule`，其他示例为 `triton_rule`；绑定/编译/搜索成本另列，不能忽略为免费启动。Graph 使用静态地址契约；同一 Attention 在两种协议下结论不同。Transformer 的兼容编译基线含已披露的外部 GEMM 边界。以上未作统计显著性认证，不能概括成项目整体加速率。

## 本地环境与快速开始

所有命令在本目录执行。使用项目独立 Python 3.12 / `.venv`，不安装到 macOS 系统 Python，也不复用课程环境。

```sh
uv sync --frozen
uv run --frozen triton-lab doctor
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen python -m compileall -q src scripts tests
uv run --frozen triton-lab check --device cpu
uv run --frozen triton-lab bench --config configs/smoke.json --dry-run
```

`uv.lock` 固定实际版本。PyTorch 固定为 2.8.0；Linux x86_64 的依赖包含其匹配的 Triton 和 CUDA 库，Mac arm64 使用原生 wheel，不安装 Triton。GPU 驱动仍由远程主机提供，不能由 Python 环境替代。项目支持矩阵为 macOS arm64 和 Linux x86_64。

uv 沿用工作区既有共享缓存；不手动删除缓存内部文件。独立环境只包含项目自己的依赖，重建使用 `uv sync --frozen`。

## 目录与产物

| 位置 | 内容 | Git |
|---|---|---|
| `src/triton_ops_lab/` | 接口、参考、GPU 草案、模型、实验与分析 | 提交 |
| `tests/`、`configs/` | 正确性、工具测试、独立实验集 | 提交 |
| `docs/`、`notebooks/`、`scripts/` | 中文资料、无输出 Colab 入口、远程工具 | 提交 |
| `.venv/`、测试/工具缓存 | 项目本地环境及可再生缓存 | 忽略 |
| 工作区 `storage/runs/triton/` | 运行指标、日志、报告、图表、profiling 和编译产物 | 不提交 |
| 工作区 `storage/datasets/triton/`、`storage/checkpoints/triton/` | 将来需要的数据和权重 | 不提交；当前不下载训练数据 |
| `local-data/downloads/` | 已从浏览器下载目录收回的实验副本及私有哈希清单 | 忽略，不上传 |
| `local-data/publish/` | 经过脱敏的 GitHub 分享副本与检查记录 | 忽略；仅单独发布其 `repository/` 源码 |

输出优先级：`--output-root` > `TRITON_LAB_OUTPUT_ROOT` > 根据本仓库位置定位工作区 `storage/runs/triton/`。独立克隆找不到工作区存储时必须显式指定。拒绝把输出根目录配置到源码仓库内；每次运行有独立时间戳和随机标识，不覆盖旧结果。小型单元测试使用临时目录。

仓库只在本目录初始化，上层 `AI-Infra` 不是 Git 仓库。原工作仓库保留本地审计历史；GitHub 使用另行检查的脱敏快照和干净提交历史，其远程命名 `origin`。本实验没有课程官方仓库，无需 `upstream`。详见[整理与分享约定](docs/project-layout.md)。

## Python 接口

```python
import torch
from triton_ops_lab import matmul, attention, conv2d, softmax, vector_add

a, b = torch.randn(17, 19), torch.randn(19, 23)
y = matmul(a, b, backend="reference")
```

- `backend="torch"` 为默认框架路径；`reference` 为可读算法；`triton` 为 GPU 实验实现，显式失败而非自动回退。`torch_mm_compat` 仅用于明确命名的 `compile_mm_compat` 兼容基线，外部 GEMM 边界会被审计记录。
- 全部接口只用于前向推理，拒绝 `requires_grad=True` 输入。需要训练的用户必须另外实现并验证 backward。
- GEMM 接收 `[M,K] @ [K,N]`，正步长连续/转置布局；Bias 为 `[N]`，`activation="gelu"` 指 tanh 近似。`variant` 为 `basic/grouped/fused`；非融合变体的 FP32 epilogue 中间张量计入耗时和显存。
- Attention Q/K/V 为同形状连续 `[B,H,N,D]`，`D=32/64/128`；`causal=True` 包含对角线；无 dropout/GQA/KV cache。
- Conv2D 是 NCHW/OIHW cross-correlation，核 1×1/3×3、stride 1/2、padding 0/1、groups=dilation=1；直接处理 NCHW，没有隐形 im2col 或格式转换。
- softmax 为二维连续输入，按最后一维归约；GPU 草案最多 16384 列。向量加法输入为同形状连续一维向量。
- CPU 参考支持 FP32/FP64 及低精度输入转 FP32 累加；GPU 仅 FP16/BF16，要求 Linux NVIDIA CC>=8.0。
- `config` 用于显式 GPU 分块配置，实验系统负责候选选择；运行记录必须保存选中的配置。
- `prepare_matmul` 为重复使用相同输入张量的 GEMM 预绑定接口。允许原地更新内容；每次检查元信息与存储，输出仍新分配。更换输入对象需要重新绑定，Graph 重放仍要求静态地址契约。

## 实验与分析

Mac 不执行正式 GPU benchmark。`bench --dry-run` 不创建运行目录，不生成性能值。

```sh
uv run --frozen python scripts/synthetic_demo.py
# 用上一步打印的绝对目录替换 <run>；只有测试报告才允许 --allow-synthetic
uv run --frozen triton-lab analyze <run> --allow-synthetic
```

合成数据、图标题和报告明确标为 SYNTHETIC；默认分析器拒绝它们。真实运行不使用该开关。CLI 返回码：0 成功、1 正确性/实验验收失败、2 输入/环境/执行错误。GPU pytest 跳过不能算 GPU 验收通过；远程应先执行 `check --device cuda`。

## 文档入口

- [后续工作与优先级](docs/next-steps.md)
- [整理与分享约定](docs/project-layout.md)
- [数据与历史版本可用性](docs/data-availability.md)
- [完整任务与验收范围](docs/plan.md)
- [一年路线与预算](docs/roadmap.md)
- [算法与 GPU/Triton 基础](docs/theory.md)
- [实验、误差和结果协议](docs/protocol.md)
- [Linux / Colab 与 profiling 交接](docs/remote.md)
- [来源与文献矩阵](docs/sources.md)
- [本地验收记录](docs/acceptance.md)
- [首轮 L4 验证](docs/colab-validation-20261020.md)
- [编译与计时审计](docs/baseline-audit-20261020.md)
- [编译兼容与融合消融](docs/mm-compat-validation-20261021.md)
- [GEMM 搜索与主机开销验证](docs/gemm-tuning-validation-20261021.md)
- [内存安全、编译产物与硬件资源验证](docs/gpu-resource-validation-20261021.md)

本项目不预设超越 cuBLAS、cuDNN 或融合 SDPA。性能结论必须由真实测量、强基线和退化分析支持。
