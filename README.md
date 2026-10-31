# Triton Ops Lab

基于 Triton 实现通用矩阵乘法（GEMM）、融合注意力（Attention）和二维卷积（Conv2D），研究分块调度与算子融合的性能收益。以 PyTorch 及其编译、融合注意力实现为基线，在 NVIDIA L4 和 A100 上进行正确性检查与性能评测。

[性能结果](#代表性性能结果) · [快速开始](#快速开始) · [完整项目文档](docs/project.md)

## 核心能力与优化方法

GEMM 采用分块计算和分组调度，支持非整除尺寸，并将偏置与 GELU（高斯误差线性单元）激活融合到乘加结果中。GELU 使用 tanh 近似；预绑定接口预先固定输入张量与内核配置，以复用调度参数。

Attention 通过在线 Softmax 递推完成融合前向计算，避免将完整注意力矩阵写入显存。Conv2D 采用隐式 GEMM 计算输入地址，支持范围见[接口说明](docs/project.md#1-范围与运行环境)。

根据开发集测量结果建立形状选参策略，在独立评测前固定。TinyCNN 和 TinyTransformer 用于检验优化在完整模块前向中的效果；消融实验、编译缓存成本及硬件计数器用于分析优化收益与开销。

## 代表性性能结果

实验采用 16 位浮点数（FP16），运行于 NVIDIA L4。会话 A 和会话 B 为两个独立 GPU 会话，每个数值为相应会话的配对加速比中位数。加速比为基线耗时除以 Triton 耗时：1 表示持平，大于 1 表示加速，小于 1 表示减速。

| 案例与输入形状 | 基线 | 计时方式 | 加速比（会话 A） | 加速比（会话 B） |
|---|---|---|---|---|
| 预绑定 GEMM + Bias + GELU，`[M,N,K]=[129,193,257]` | PyTorch eager | CUDA Event | 2.026× | 2.013× |
| 非因果 Attention，`[B,H,N,D]=[1,2,65,32]` | SDPA Flash | CUDA Graph | 1.636× | 1.640× |
| 同一 Attention | SDPA Flash | CUDA Event | 0.858× | 0.869× |
| Conv2D，`[B,C,H,W,OC,kernel]=[1,16,32,32,32,3]`，步长 1，填充 1 | PyTorch eager | CUDA Graph | 3.144× | 3.142× |
| TinyCNN 完整前向，`[B,C,H,W]=[1,3,31,29]` | torch.compile | CUDA Graph | 1.453× | 1.474× |
| TinyTransformer 完整前向，`[B,N,width]=[1,127,64]` | compile_mm_compat | CUDA Graph | 0.908× | 0.910× |

在所列输入上，预绑定融合 GEMM 的延迟约降低一半。CUDA Graph 下，Conv2D 和 TinyCNN 获得加速，TinyTransformer 则慢于编译兼容基线。同一 Attention 在 CUDA Graph 下快于 SDPA Flash，在 CUDA Event 计时下慢于该基线。

PyTorch eager 为即时执行基线，SDPA Flash 为 PyTorch 缩放点积注意力的 Flash 后端。GEMM 使用预绑定路径 `triton_prepared_rule`，其余案例使用 `triton_rule`；`compile_mm_compat` 是包含 FP32 输出 GEMM 外部调用边界的编译兼容基线。表中的 Bias 指偏置，输入形状沿用[实验配置的维度约定](docs/project.md#4-实验设计与计时)。

CUDA Event 包围重复调用进行计时；CUDA Graph 固定输入地址，捕获调用后测量重放时间。结果按输入形状和计时方式分别解释，Graph 重放结果不用于表示普通 API 调用延迟。完整定义见[实验协议](docs/project.md#4-实验设计与计时)和[结果说明](docs/project.md#5-验证覆盖与结果)。

### 实验覆盖

独立评测包含两个 L4 会话和一个 A100 补充会话，结果分别统计。开发实验包含 13 个 GEMM 逐因素配置和 384 组消融比较，缓存实验包含 42 对冷缓存与热缓存测量。正确性测试覆盖 CPU 参考实现与 GPU 内核，详细范围见[验证覆盖](docs/project.md#5-验证覆盖与结果)。

完整应用的内存初始化与竞争检查仍有厂商内核告警，项目范围检查及例外规则见[检查边界](docs/project.md#安全与硬件解释的边界)。

## 快速开始

在项目根目录使用 Python 3.12，通过 uv 创建独立的 `.venv` 环境：

```sh
uv sync --frozen
uv run --frozen triton-lab doctor
uv run --frozen pytest -q
uv run --frozen triton-lab check --device cpu --output-root /tmp/triton-results
uv run --frozen triton-lab bench --config configs/smoke.json --dry-run
```

macOS 支持代码编辑、CPU 验证和数据分析。GPU 实验要求 Linux x86_64 和计算能力 8.0 及以上的 NVIDIA GPU，依赖锁定为 PyTorch 2.8.0 和 Triton 3.4.0；Triton 仅在 Linux 环境安装。

实验结果写入源码目录之外的位置，通过 `TRITON_LAB_OUTPUT_ROOT` 或 `--output-root` 指定。GPU 和 Colab 执行步骤见[复现指南](docs/project.md#6-在-nvidia-linux--colab-上复现)。

## 项目结构

| 路径 | 内容 |
|---|---|
| `src/triton_ops_lab/` | 接口、CPU 参考、Triton 内核、模型与实验分析 |
| `configs/`、`tests/` | 实验计划、固定策略和验证 |
| `scripts/`、`notebooks/` | 本地与 GPU 执行入口 |
| `docs/project.md` | 完整项目文档 |
| `tools/` | 发布检查工具 |

接口仅用于前向推理，不支持的输入或梯度请求返回错误。原始测量与完整日志保存在仓库外的实验归档中。
