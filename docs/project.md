# Triton 算子优化：实现、实验与复现

本项目研究 GEMM、融合 Attention 与受限 Conv2D 的分块、访存、调度和融合，并用一致的正确性门禁、计时协议和框架基线检验优化效果。核心方法是从开发集建立按形状选参的固定策略，在独立形状及完整模块上验证收益与退化，同时记录搜索、编译缓存和资源成本。

## 1. 范围与运行环境

接口用于前向推理，拒绝 `requires_grad=True`；不包含 backward、训练、GQA、KV cache、多卡或完整手写 CUDA 对照。向量加法和 softmax 用于基础验证，性能研究围绕三类核心算子及 TinyTransformer、TinyCNN 展开。

| 算子 | 输入与支持范围 |
|---|---|
| GEMM | `[M,K] @ [K,N]`，连续或转置形成的正步长布局；可选列 Bias、tanh 近似 GELU；支持非整除尺寸 |
| Attention | 连续且同形状的 Q/K/V `[B,H,N,D]`，`D=32/64/128`；causal 包含对角线；无 dropout |
| Conv2D | NCHW/OIHW cross-correlation，核 1×1/3×3、stride 1/2、padding 0/1、groups=dilation=1 |
| Softmax | 二维连续输入，沿末维归约；GPU 最多 16384 列 |
| Vector add | 同形状连续一维输入 |

GPU 使用 FP16/BF16，要求 Linux x86_64、NVIDIA CC≥8.0。CPU 参考支持 FP32/FP64，低精度输入提升到 FP32 累加。Mac arm64 负责编辑、CPU 验证和数据分析，不执行 CUDA/Triton 性能测量，也不以 MPS 替代 CUDA 验证。

所有命令在项目根目录执行，使用项目独立的 Python 3.12 和 `.venv`：

```sh
uv sync --frozen
uv run --frozen triton-lab doctor
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen triton-lab check --device cpu --output-root /tmp/triton-results
uv run --frozen triton-lab bench --config configs/smoke.json --dry-run
```

`uv.lock` 固定 PyTorch 2.8.0 及依赖；Linux 使用匹配的 Triton 3.4.0 / CUDA 12.8 用户态库，Mac 不安装 Triton。GPU 驱动由远程平台提供。重建环境使用 `uv sync --frozen`，不安装到系统 Python，也不复用其他课程环境。

## 2. 实现与优化方法

Triton program 处理张量块，不等同于单个 GPU 线程。`tl.arange` 表达索引，`tl.load/store` 的 mask 保护边界，`tl.dot` 表达分块乘加；lane/warp 的具体映射由编译器及后端决定。块大小、`num_warps` 和 `num_stages` 共同影响复用、并行度、寄存器与共享内存，不能单凭占用率判断快慢。

### GEMM

`C[m,n] = Σ_k A[m,k]B[k,n]`，计算量约 `2MNK`。输出 tile 复用 A/B 载入；增大 tile 可能减少重复访问，也可能增加边界浪费与资源压力。basic 使用 GROUP=1，grouped 使用 GROUP=8，fused 在累加器内执行 Bias/GELU，避免非融合路径的 FP32 中间结果写回与读取。

GELU 采用 `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`。框架 epilogue 对照同样使用低精度输入、FP32 GEMM 输出及中间运算，保持精度语义一致。

`prepare_matmul` 预绑定具体输入张量和配置，以复用校验、grid 与常量参数。每次调用仍检查地址、形状、stride、dtype、device 和梯度状态，并分配新的输出。允许原地更新输入值；更换输入对象须重新绑定。该契约单独标记为 `triton_prepared*`，不与普通接口混称。

### Attention

标准形式为 `softmax(QKᵀ/sqrt(D))V`。融合实现按 Q 行块扫描 K/V，保留行最大值 m、归一化分母 l 和输出累加器 a，避免完整 N×N scores/probabilities 落显存：

```text
m_new = max(m, rowmax(S))
alpha = exp(m - m_new)
p = exp(S - m_new)
l_new = alpha*l + rowsum(p)
a_new = alpha*a + p@V_block
output = a/l
```

实数下该递推与整体 softmax 等价；浮点运算顺序和低精度 dot 仍会引入误差。causal mask 屏蔽未来位置并跳过完全无用的块。实现属于基础融合前向，不代表实现了 FlashAttention-2 的全部优化，也不意味着优于成熟融合 SDPA。

### Conv2D

将 `(channel,kh,kw)` 作为归约维，将 batch 和输出空间位置作为 M、输出通道作为 N，直接计算 implicit GEMM 地址：

```text
input_y = output_y*stride - padding + kh
input_x = output_x*stride - padding + kw
Y[b,oc,oy,ox] = Σ X[b,c,input_y,input_x]*W[oc,c,kh,kw]
```

越界输入贡献零，不隐藏 im2col 或格式转换。NCHW gather 和地址计算的开销可能抵消 Tensor Core 收益，性能须按具体形状判断。

### 固定形状策略

开发集按 FP16/BF16 分开拟合近邻启发式：取最多三个开发邻居，将候选成本在各开发形状内归一化，再按距离加权。静态路径选择排名第一项；短名单路径保留前两项和默认配置，最多三项。特征、距离、参数和候选池在独立评测前冻结，策略保存特征源码与开发数据哈希。

该方法用于检验候选筛选的收益，不声称提出新的近邻学习算法。独立集不参与策略拟合；评测中在线搜索的后端须单独记录调参成本。见 `configs/policy-v1.json`、`configs/study/policy-decisions-v1.json` 与 `src/triton_ops_lab/policy.py`。

## 3. 接口与目录

```python
import torch
from triton_ops_lab import matmul

a, b = torch.randn(17, 19), torch.randn(19, 23)
y = matmul(a, b, backend="reference")
```

`torch` 为默认框架路径，`reference` 为可读数学实现，`triton` 为 GPU 实现；不支持的输入明确失败，不静默回退。GEMM 的 `variant` 为 `basic/grouped/fused`，`config` 可指定分块参数。

| 路径 | 职责 |
|---|---|
| `src/triton_ops_lab/` | 接口、参考算法、GPU 内核、模型、实验与分析 |
| `tests/`、`configs/` | 正确性检查、候选、固定策略和实验计划 |
| `scripts/`、`notebooks/colab.ipynb` | 本地/远程执行、编译产物采集和实验入口 |
| `docs/project.md` | 项目完整说明 |
| `tools/` | 源码发布检查工具 |
| `.venv/`、缓存 | 可重建环境，不纳入 Git |
| `local-data/` | 本地实验附件与管理记录，不纳入 Git |

结果路径优先级为 `--output-root`、`TRITON_LAB_OUTPUT_ROOT`、工作区 `storage/runs/triton/`。独立克隆必须显式设置外部输出位置。运行目录含时间戳与随机标识，不覆盖旧记录；拒绝把 benchmark 输出放进源码仓库。

## 4. 实验设计与计时

### 正确性

框架与自定义实现共用同一实际输入和随机种子。低精度 oracle 从已量化输入提升到 FP32，不重新生成输入；CPU 数学验证另用 FP64。GEMM/Conv2D 使用框架 FP32 oracle，Attention 使用显式 FP32 数学形式，模块比较相同权重与低精度边界的完整前向。

要求输出形状一致、全值有限，并逐元素满足 `abs(actual-reference) <= atol + rtol*abs(reference)`：

| dtype | 默认 atol / rtol | Softmax | Attention |
|---|---|---|---|
| FP64 | 1e-9 / 1e-8 | 同默认 | 同默认 |
| FP32 | 2e-5 / 2e-4 | 同默认 | 同默认 |
| FP16 | 2e-2 / 2e-2 | 2e-3 / 1e-2 | 8e-3 / 2e-2 |
| BF16 | 8e-2 / 5e-2 | 8e-3 / 3e-2 | 4e-2 / 5e-2 |

FP32 Transformer 使用 1e-4 / 5e-4。阈值以 `validation.py` 为准，不为通过实验而自动放宽。额外记录 max_abs、relative_l2 和带稳定分母的 max_scaled_relative；诊断指标不代替逐元素门禁。

### 配置与基线

JSON shape 顺序：GEMM `[M,N,K]`、Attention `[B,H,N,D]`、Conv2D `[B,C,H,W,OC,kernel]`、Transformer `[B,N,width]`、CNN `[B,C,H,W]`。smoke 检查边界；开发集用于调优和消融；独立集包含 24 个算子/模块案例。修改规则后不能继续把已经使用的数据称为未见集合。

- `torch`：相同语义的 eager 框架对照。
- `compile`：`torch.compile(fullgraph=True, dynamic=False)`；Inductor 内部 CUDA Graph 关闭，要求编译审计得到单图。单图不等于所有算子都融合，仍检查实际调度。
- `compile_mm_compat`：FP32 输出 GEMM 的兼容对照。注册的自定义算子仍调用 `torch.mm(..., out_dtype=torch.float32)`；Inductor 将其视为外部边界，可以编译周围运算。它不代表原生 `aten.mm.dtype` 编译问题已修复，原生失败仍保留。
- `sdpa_math/flash/efficient`：强制相应 Attention 后端，不支持或失败时保存原因，不回退数学路径。
- `triton_basic/grouped/fused`、`triton_fixed/search/pruned`、`triton_prepared*`、`triton_rule*`：分别比较融合、候选筛选、绑定契约及冻结策略。只在支持的算子上使用。

TF32、FP16/BF16 reduced-precision reduction、cuDNN benchmark 按正式计时协议关闭。输入传输、oracle 和正确性预检不计入算子延迟；算子必要的分配与转换计入。

### 两种协议

`cuda_event` 使用 CUDA events 包围重复调用，同时保存同步墙钟样本；结果除以重复次数。短内核的 event 区间可能包含主机提交间隙，不能称为纯指令耗时。

`cuda_graph` 预热后捕获静态地址的重复调用，对 replay 计时，重放输出再次通过 oracle。它减少重复 Python 调用和分配开销，不能替代普通 API 延迟。显存与内核名称仍来自捕获之外的普通调用，不表示 Graph 私有池占用。

每种协议执行三个重复测量任务并轮换后端顺序。样本保留 p20/median/p80/min/max、event 与 wall 原值；GPU、dtype、协议、案例及独立会话不混合统计。选参后独立重测，不以搜索最小值作为最终结果。

`first_call_ms` 是同步首次调用墙钟，包含编译/加载/执行；`construction_ms` 是工厂构造成本；`tuning_ms` 包含候选构造、首次调用、正确性和搜索采样。冷/热实验控制 Triton/Inductor 文件缓存，不等于 GPU 驱动和整个进程的完全冷启动。离线开发采样与 CPU 拟合分开核算，避免重复相加；无稳定延迟收益时不承诺正向摊销。

显存记录 PyTorch allocator 峰值及相对调用前增量，不等同于全部驱动/第三方分配或 nvidia-smi 数值。FLOPs 统计按 GEMM 的 2MNK、卷积有效输出及 Attention 两次矩阵乘法计算；不计全部标量操作，模块不编造 FLOPs。

## 5. 验证覆盖与结果

历史 GPU 实验使用固定版本 `54f29c2`。已完成开发、冻结策略、独立评测、同型号独立会话复验及 A100 补充验证。

| 验证项 | 已核验覆盖 |
|---|---|
| Mac 本地测试 | 156 项通过、96 项 GPU 测试按环境要求跳过 |
| GPU 正确性 | L4 与 A100 均有 28 项 CUDA 检查、96 项 GPU 测试通过 |
| 开发与消融 | 20 个任务、1,488 条记录；13 个 GEMM 逐因素配置、384 组消融比较 |
| 独立评测与复验 | 两个 L4 会话各 30 个任务、1,284 条记录，含完整模块前向 |
| A100 补充 | 单会话 30 个任务、1,284 条记录；沿用 L4 冻结策略，无额外拟合 |
| 编译缓存成本 | 84 个任务、42 对冷/热缓存 |
| 编译与硬件 | 16 组 IR/二进制/SASS、6 个 Nsight launch |

以下示例均为 FP16，两个数值分别来自两个独立 L4 会话的配对加速比中位数：

| 案例 | 协议 | 基线 | 加速比 | 延迟减少 |
|---|---|---|---|---|
| 预绑定 GEMM+Bias+GELU，M/N/K=129/193/257 | event | torch | 2.026× / 2.013× | 50.64% / 50.33% |
| 非 causal Attention，B/H/N/D=1/2/65/32 | graph | sdpa_flash | 1.636× / 1.640× | 38.89% / 39.01% |
| 同一 Attention | event | sdpa_flash | 0.858× / 0.869× | -16.61% / -15.12% |
| Conv2D，1/16/32/32/32/3，stride=1、padding=1 | graph | torch | 3.144× / 3.142× | 68.19% / 68.17% |
| TinyCNN，输入 1/3/31/29 | graph | compile | 1.453× / 1.474× | 31.18% / 32.14% |
| TinyTransformer，B/N/D=1/127/64 | graph | compile_mm_compat | 0.908× / 0.910× | -10.12% / -9.83% |

加速比为基线耗时除以 Triton 耗时，延迟减少为 `100×(1−1/加速比)`；负值表示退化。GEMM 使用 `triton_prepared_rule`，其余示例为 `triton_rule`。这些是指定契约下的示例，不是项目整体加速率，也没有作统计显著性认证。跨会话比较保留各自的全部重复值及方向变化。

A100 只有一个完整会话，不与 L4 混算，也没有重复 L4 的安全与性能计数器采集。不同形状的收益并不一致；单算子优势不能直接当作完整模型优势。

### 安全与硬件解释的边界

L4 的项目范围检查包含 memcheck、synccheck、racecheck 与全部地址空间 initcheck，但完整应用的原始 racecheck/initcheck 仍有厂商内核告警。racecheck 使用已归属的精确厂商内核名排除；initcheck 保留所有项目内核及输入生产者插桩，只按框架独立复现得到的字面函数、16 字节共享读取和指令位置匹配报告例外。

例外由正对照、故意未初始化的全局读取及六个项目内核名下的同类型共享读取故障检出验证约束。未打印的原始事件不能归属为安全事件；不能宣称告警是误报、厂商问题已解决，或整个框架已经通过无告警认证。仅筛选读取者可能丢失生产者跟踪，不能据此认证初始化安全。

Nsight 与编译产物用于关联块配置、寄存器、共享内存、实际指令及访存计数器；插桩耗时不计入性能结论。计算指标保留实际采集的 `sm__instruction_throughput.avg.pct_of_peak_sustained_elapsed`，不替换成另一项吞吐指标。温度、频率和功耗只是运行端点观测，未锁定时钟，也不是逐样本跟踪。没有直接证据时只提出瓶颈假设。

## 6. 在 NVIDIA Linux / Colab 上复现

检出明确提交，保持锁文件不变，设置源码仓库之外的持久化输出路径：

```sh
export TRITON_LAB_OUTPUT_ROOT=/your/persistent/storage/triton
uv sync --frozen
uv run --frozen triton-lab doctor
uv run --frozen triton-lab check --device cuda
uv run --frozen python scripts/gpu_tools.py check
uv run --frozen triton-lab bench --config configs/smoke.json
```

正确性失败立即停止。确认 GPU 型号、驱动、torch/Triton 版本和 BF16 支持，再运行正式实验；Colab 全局预装包不能替代 uv 项目环境。T4 不满足本项目主线的计算能力要求。凭据使用平台正常认证机制，不写入 URL、代码或输出。

实验命令如下；冻结操作仅用于新的开发实验，不能覆盖已有策略后继续使用原评测结论：

```sh
uv run --frozen python scripts/run_study.py --suite configs/study/development-suite.json
uv run --frozen python scripts/freeze_policy.py <development-run> --output <new-policy.json>
uv run --frozen python scripts/run_study.py --suite configs/study/evaluation-suite.json
uv run --frozen python scripts/run_study.py --suite configs/study/replication-suite.json
uv run --frozen python scripts/run_study.py --suite configs/study/cost-suite.json
```

独立克隆需显式设置上述输出环境变量或 `--output-root`。`notebooks/colab.ipynb` 调用同一源码，不维护另一套内核；`scripts/build_colab_delivery.py --suite study-<phase>` 可从干净提交生成含固定源码 bundle 的 notebook 和哈希清单。

```sh
uv run --frozen python scripts/gpu_tools.py sanitize
uv run --frozen python scripts/gpu_tools.py capture --case gemm-epilogue-float16
uv run --frozen python scripts/gpu_tools.py profile --case gemm-epilogue-float16
uv run --frozen python scripts/qualify_gpu.py all
```

采集可用的 TTIR/TTGIR/LLVM IR/PTX/cubin，再用兼容工具读取 SASS；正式 benchmark 拒绝在 capture 环境变量开启时运行。ncu 权限不足、工具缺失、超时、退出异常或未生成产物均不能视为通过，不通过修改系统安全配置绕开权限。插桩子进程可使用 `TORCHINDUCTOR_COMPILE_THREADS=1` 处理编译退出等待，其设置需记录，不能暗改正式计时策略。

## 7. 结果管理与审计

运行目录保存 `metadata.json`、`metrics.csv`、`preflight.json` 和候选/编译记录。状态区分 passed、incorrect、failed、unsupported；中断保留 `complete=false`。原始样本、选中配置、失败与退化不能覆盖或删除。

每个 L4 会话有六条 Attention 的独立 profiler 调用缺少内核名称。这些空记录保持不变，由相同源码、完整环境、种子和已选配置的不计时 GPU 轨迹补充。补采不重新搜索、不替换计时；保存脚本、原 CSV/metadata 哈希和全部尝试，最多三次并保留首次成功观测。审计不接受错误字符串或不同配置借补采通过。

原始结果、完整日志和执行 notebook 保存在本地实验归档，不包含在仓库中；文档摘要不能代替对原始样本的独立核验。历史实验关联其原始源码 bundle 和 manifest，重新运行应记录新提交与环境，不替换历史记录。数据来源及未验收的中断尝试保留在归档中。

```sh
uv run --frozen python scripts/audit_completion.py \
  --development <development-run> --evaluation <evaluation-run> \
  --replication <replication-run> --cost <cost-run> \
  --hardware <qualification-manifest> --output <external-audit-directory>
```

完成审计核对覆盖、样本中位数、候选正确性、实际调度、单图编译、冻结决策、独立 L4 身份、缓存配对和工具日志。`replication-consistency.json` 描述观测方向一致性，不是显著性检验；逐案例比较不合并协议、精度或会话。历史审计应在对应的原始源码环境执行，文档路径维护不代表重新测量了 GPU 性能。

Colab `/content` 不是长期存储。下载后核对字节数、SHA256、ZIP 完整性和路径，再释放自己的运行时并确认消耗归零。仅显示导出成功、停止单元格或断开 SSH 都不等于归档成功或停止计费。可以用 notebook 输出保存原始包字节检查点作为额外备份；真实计时与配置仍保持不变。

合成流程使用 `uv run --frozen python scripts/synthetic_demo.py`，分析时须显式 `--allow-synthetic`；图表必须注明 SYNTHETIC / TEST DATA，不能进入真实性能结论。普通分析使用 `uv run --frozen triton-lab analyze <run>`。CLI 返回码 0 为成功、1 为正确性/实验失败、2 为输入或环境错误；Mac 上 GPU 跳过不是 GPU 验收证据。

## 8. 后续研究

优先分析已有结果的适用范围、主机提交开销与模块瓶颈，明确方法贡献与已有方法的关系。若论证暴露具体证据缺口，再增加特定形状重复测量或资源诊断。修改内核、精度、依赖或配置须另立版本并重新验证；backward、GQA、KV cache 和多卡属于额外研究范围。

## 9. 参考资料与来源声明

| 来源 | 本项目用途 |
|---|---|
| [Triton 教程](https://triton-lang.org/main/getting-started/tutorials/)、[GEMM](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)、[融合 Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) | 张量块编程、分组映射与融合实现参考 |
| [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) | 在线归一化递推 |
| [FlashAttention](https://arxiv.org/abs/2205.14135)、[FlashAttention-2](https://arxiv.org/abs/2307.08691) | IO-aware attention、并行划分与贡献边界 |
| [torch.mm 2.8](https://docs.pytorch.org/docs/2.8/generated/torch.mm.html)、[Attention 后端](https://docs.pytorch.org/docs/2.8/nn.attention.html) | 精度语义与明确选择后端 |
| [PyTorch fullgraph](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html)、[CUDA Graph](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)、[自定义算子](https://docs.pytorch.org/tutorials/advanced/python_custom_ops_registrations.html) | 编译与计时契约 |
| [uv/PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/) | 平台依赖管理 |
| [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)、[Nsight CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) | 插桩检查与硬件计数器 |
| [Triton 3.4 编译器](https://github.com/triton-lang/triton/blob/v3.4.0/python/triton/compiler/compiler.py) | 编译产物与资源字段 |

分块 GEMM、分组调度、在线 softmax、融合 Attention 和 implicit GEMM 都是已有方法。GEMM 分组映射与 Triton 教程具有直接衍生关系，向量索引、softmax 和 Attention 模式也受官方示例启发。下列上游 MIT 声明适用于相关来源/衍生片段，不替其他依赖重新授权，也不表示上游作者对本项目背书；本项目未另行新增使用许可证。

[Triton v3.4.0 原始许可](https://github.com/triton-lang/triton/blob/v3.4.0/LICENSE)：

```text
Copyright 2018-2020 Philippe Tillet
Copyright 2020-2022 OpenAI

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files
(the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
