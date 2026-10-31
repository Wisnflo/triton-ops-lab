# Triton Ops Lab

**从 GPU 算子实现到完整模块验证的 Triton 优化研究。**

基于 Triton 实现分块 GEMM、融合 Attention 与隐式 GEMM 卷积，探索算子融合、调度及形状选参如何影响实际性能，并在 NVIDIA L4 / A100 上与 PyTorch、编译基线和融合 SDPA 对照。

`Python 3.12` · `PyTorch 2.8.0` · `Triton 3.4.0` · `NVIDIA L4 / A100`

[性能结果](#代表性性能结果) · [快速开始](#快速开始) · [完整项目文档](docs/project.md)

## 核心能力与优化方法

- **实现三类核心算子**：支持非整除尺寸的 GEMM、Bias / tanh GELU 融合、在线 Softmax Attention，以及受限 Conv2D；提供 CPU 数学参考与统一接口。
- **比较优化策略的实际收益**：研究分块、分组调度、融合与预绑定调用；从开发集建立冻结的形状选参策略，在独立案例上评测。
- **验证完整模块与优化成本**：覆盖 TinyCNN / TinyTransformer 前向，同时记录正确性、搜索与编译缓存成本，并结合消融、编译产物和硬件计数器分析结果。

## 代表性性能结果

以下均为 **FP16、NVIDIA L4** 上指定输入的测量结果；两个加速比分别为两个独立会话的配对加速比中位数。

| 案例与输入形状 | 基线 | 计时方式 | 加速比：会话 A / B |
|---|---|---|---|
| 预绑定 GEMM + Bias + GELU，M/N/K = 129/193/257 | PyTorch eager | CUDA Event | **2.026× / 2.013×** |
| 非 causal Attention，B/H/N/D = 1/2/65/32 | SDPA Flash | CUDA Graph | **1.636× / 1.640×** |
| 同一 Attention | SDPA Flash | CUDA Event | 0.858× / 0.869× |
| Conv2D，B/C/H/W/OC/kernel = 1/16/32/32/32/3，stride = 1、padding = 1 | PyTorch eager | CUDA Graph | **3.144× / 3.142×** |
| TinyCNN 完整前向，B/C/H/W = 1/3/31/29 | torch.compile | CUDA Graph | **1.453× / 1.474×** |
| TinyTransformer 完整前向，B/N/width = 1/127/64 | compile_mm_compat | CUDA Graph | 0.908× / 0.910× |

加速比 = 基线耗时 / Triton 耗时；小于 1 表示退化。GEMM 使用预绑定路径 `triton_prepared_rule`，其余案例使用 `triton_rule`。`compile_mm_compat` 是包含 FP32 输出 GEMM 外部边界的编译兼容基线。

CUDA Graph 使用静态地址捕获与重放，不能代替普通 API 调用延迟。结果取决于形状、精度、调用契约和计时方式，不代表所有输入或完整模型都能获得同样收益。完整定义见[实验协议](docs/project.md#4-实验设计与计时)和[结果说明](docs/project.md#5-验证覆盖与结果)。

### 实验覆盖

| 维度 | 已完成验证 |
|---|---|
| GPU 评测 | 两个独立 L4 会话、一个 A100 会话，各 30 个任务、1,284 条记录 |
| 开发与消融 | 13 个 GEMM 逐因素配置、384 组消融比较 |
| 缓存成本 | 42 对冷 / 热缓存实验 |
| GPU 正确性 | L4 与 A100 均通过 28 项 CUDA 检查及 96 项 GPU 测试 |
| 本地验证 | 156 项测试通过；96 项 GPU 测试按环境要求跳过 |

A100 结果作为单会话补充，未与 L4 混合统计。完整应用仍保留厂商库安全告警，项目范围检查的具体边界见[项目文档](docs/project.md#安全与硬件解释的边界)。

## 快速开始

使用 Python 3.12 和项目独立 uv 环境：

```sh
uv sync --frozen
uv run --frozen triton-lab doctor
uv run --frozen pytest -q
uv run --frozen triton-lab check --device cpu --output-root /tmp/triton-results
uv run --frozen triton-lab bench --config configs/smoke.json --dry-run
```

Mac 用于编辑、CPU 验证和分析；真实 GPU 实验要求 Linux x86_64、NVIDIA CC≥8.0，使用锁定的 PyTorch 2.8.0 / Triton 3.4.0。独立克隆须将 `TRITON_LAB_OUTPUT_ROOT` 或 `--output-root` 指向源码目录之外。

GPU / Colab 执行步骤与完整实验命令见[复现指南](docs/project.md#6-在-nvidia-linux--colab-上复现)。

## 项目结构

| 路径 | 内容 |
|---|---|
| `src/triton_ops_lab/` | 接口、CPU 参考、Triton 内核、模型与实验分析 |
| `configs/`、`tests/` | 实验计划、固定策略和验证 |
| `scripts/`、`notebooks/` | 本地与 GPU 执行入口 |
| `docs/project.md` | 完整项目文档 |
| `tools/` | 发布检查工具 |

所有接口只支持前向推理，明确拒绝未支持的输入与梯度请求，不静默回退。原始测量与完整工具日志保存在实验归档中，不包含在仓库内。

实现原理、接口约束、完整实验协议、结果分析及参考资料统一收录于 **[项目文档](docs/project.md)**。
