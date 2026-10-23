# Triton 算子优化研究：Mac 本地阶段完整实施计划

> 本文保留最初本地阶段/一年路线的范围。后续授权的必做 GPU 技术实验已于 2026-10-22 通过完整验收；最新证据与限制以 [README](../README.md#技术验收与证据) 和[技术执行约定](study-workflow.md)为准，论文等提交物仍不属于本次技术交付。

## 1. 目标、范围与完成定义

在 `<local-project-path>` 建立独立研究项目，完成此前研究方案中**所有不依赖真实 NVIDIA GPU 的实现、验证与准备工作**。

账号、订阅和算力余额记录仅在本地保留。

本阶段必须交付：

- 可安装、可复现的独立 Python 项目。
- 三类算子的 CPU 参考实现及完整 Triton GPU 内核草案。
- 正确性测试、基准测试、消融、配置选择及结果分析工具。
- 小型 Transformer／CNN 前向验证模块。
- Colab／Linux GPU 执行入口、中文研究资料和验收记录。

**“本地阶段完成”不等于“研究课题完成”。** GPU 内核编译、GPU 正确性、性能收益、硬件分析和论文实验结论均需真实 GPU 验证。不得用静态检查、CPU 测试或合成数据替代这些证据。

不纳入本阶段：反向传播、模型训练、GQA、KV cache 解码、分布式计算、修改 Triton 编译器、全套手写 CUDA 对照实现。

## 2. 文件安排、Git 与 uv 环境

严格遵守根目录、`infra/environments/`、`infra/cloud/` 和 `storage/` 中的 README，不将上层工作区初始化为 Git 仓库，不改动其他课程环境。

目录安排如下：

```text
AI-Infra/
├── labs/triton/                 # 独立 Git 仓库
│   ├── README.md                # 中文上手与命令入口
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── .python-version
│   ├── .gitignore
│   ├── .venv/                   # uv 创建，不提交
│   ├── src/triton_ops_lab/
│   │   ├── references/          # CPU 参考算法
│   │   ├── kernels/             # Triton 内核，按需导入
│   │   ├── benchmarks/          # 基线、计时、配置选择
│   │   ├── analysis/            # 汇总、图表与报告
│   │   ├── models/              # Transformer／CNN 验证模块
│   │   └── cli.py
│   ├── tests/                   # CPU、GPU、CLI 与分析测试
│   ├── configs/                 # smoke、tune、eval 实验清单
│   ├── notebooks/              # Colab 引导入口
│   ├── scripts/                 # GPU 验收与 profiling 入口
│   └── docs/                    # 研究计划、原理、协议、验收记录
└── storage/
    ├── datasets/triton/         # 需要时创建
    ├── checkpoints/triton/      # 需要时创建
    └── runs/triton/             # 日志、指标、图表、编译与 profiling 产物
```

存储规则：

- 源码、测试、配置、文档、锁文件和不含输出的 notebook 纳入版本管理。
- `.venv`、缓存、构建产物、日志、数据集、检查点、凭据和原始实验结果不提交。
- 文档可以记录验收摘要及结果位置，但不复制整份运行日志进仓库。
- 实验输出目录通过 `--output-root` 或 `TRITON_LAB_OUTPUT_ROOT` 指定，命令行优先。本工作区默认使用根目录的 `storage/runs/triton/`。
- 独立克隆到其他机器后若无法定位工作区存储目录，要求显式配置，避免把输出意外写入源码目录。
- 每次运行创建独立目录，不覆盖历史结果。日常单元测试使用临时目录。

环境规则：

- 使用 Python 3.12 与 uv 创建项目本地 `.venv`，不安装依赖到系统 Python，不复用课程项目环境。
- 通过 `pyproject.toml` 声明依赖，通过 `uv.lock` 固定实际解析版本；默认开发依赖包含 pytest 与静态检查工具。
- Mac 安装原生 PyTorch、NumPy 和绘图等依赖；仅在 Linux NVIDIA 环境启用 GPU 依赖。Triton 版本与选定 PyTorch 的依赖约束一致。
- 默认算法测试使用 CPU，不将 MPS 作为 CUDA 的替代验证环境。
- 环境复现使用 `uv sync --frozen`，项目命令和测试使用 `uv run --frozen …`。
- 沿用既有 uv 共享缓存，不手动清理其内部目录。若沙箱限制缓存访问，按工具权限流程处理，不擅自改变全局配置。
- 仅初始化独立本地 Git 仓库；不创建远程仓库、不推送、不发布。后续私人远程使用 `origin`；没有官方课程仓库时不设置 `upstream`。

## 3. 算子、接口与实验工具

### 算子实现

| 模块 | CPU 可验证实现 | GPU 内核草案 |
|---|---|---|
| 入门算子 | 向量加法、稳定 softmax | 向量加法、行级融合 softmax |
| GEMM | PyTorch 基线、分块参考、Bias／GELU | 基础分块、分组调度、融合、有限配置搜索 |
| Attention | 显式注意力、分块在线 softmax | 标准 MHA 融合前向，causal／非 causal |
| Conv2D | 小规模直接卷积、PyTorch 对照 | 1×1 GEMM 路径、3×3 implicit GEMM |
| 模块验证 | 小型 Transformer、CNN 前向 | 显式替换自定义算子的执行路径 |

公共 Python 接口提供 `vector_add`、`softmax`、`matmul`、`attention`、`conv2d`，显式指定 `backend`。默认使用 `torch`，算法参考使用 `reference`，GPU 草案使用 `triton`。

- 导入包、查看帮助、运行 CPU 测试时不加载 Triton。
- 请求 `triton` 后端但设备或依赖不满足条件时明确报错，不静默回退。
- GPU 草案支持 FP16／BF16 输入、FP32 累加；CPU 验证以 FP32 为主，小规模 FP64 作高精度对照。
- 输入检查覆盖维数、形状、dtype、设备、布局和参数范围；不支持的输入明确拒绝。
- 本阶段接口用于前向推理；不隐式承诺 autograd 支持。

具体语义固定：

- **GEMM**：二维矩阵，支持连续及转置产生的常见正步长布局；可选输出列 Bias 和 tanh 近似 GELU；支持非整除尺寸。
- **Attention**：Q/K/V 同形状 `[B,H,N,D]`，`D∈{32,64,128}`；缩放为 `1/√D`，causal 使用含对角线的下三角掩码；无 dropout，支持非整除序列长度。
- **Conv2D**：NCHW 输入、OIHW 权重，核为 1×1／3×3，stride 为 1／2，padding 为 0／1，`groups=1`、`dilation=1`；不隐藏布局转换和预处理成本。
- **Softmax**：二维输入、沿最后一维归约，使用减最大值的稳定计算。

所有 GPU 内核必须有实际计算实现、边界掩码和对应测试，不能以占位函数冒充草案完成。官方教程或论文衍生实现保留来源及适用许可说明。

### 命令入口

提供以下统一入口：

```text
uv run --frozen triton-lab doctor
uv run --frozen triton-lab check --device cpu
uv run --frozen triton-lab check --device cuda
uv run --frozen triton-lab bench --config <配置> --dry-run
uv run --frozen triton-lab bench --config <配置>
uv run --frozen triton-lab analyze <运行目录>
```

- `doctor`：检查依赖、平台和设备，分别报告 Mac 开发条件与 CUDA 实验条件。
- `check`：运行指定设备的正确性验证；显式指定 CUDA 而不可用时失败退出。
- `bench --dry-run`：解析实验、校验配置、显示候选数及输出位置，不分配 GPU 或生成性能结果。
- `bench`：正式 GPU 评测；先验证正确性，再计时。失败和不支持的用例保存原因。
- `analyze`：汇总合法结果，输出 Markdown 报告及 PNG／SVG 静态图表。

### 实验协议与结果

- 建立 smoke、tune、eval 三套配置；调参和评测形状分离，并测试两者不会意外重叠。
- GEMM 覆盖方阵、长瘦矩阵、小规模及非整除形状；Attention 覆盖序列长度与 causal；卷积覆盖核尺寸、stride 和 padding。
- 对照包括 PyTorch eager、适用的 `torch.compile`、各 Triton 变体；Attention 区分数学 SDPA 与可用融合后端。
- 默认关闭 CUDA Graph；同组对照使用一致的精度、语义、输入和计时方式。
- GPU 预热、同步、多轮计时；编译、调参、稳态延迟分别记录，不把数据传输混入纯算子延迟。
- 保存运行环境 JSON 和逐用例 CSV：代码版本及 dirty 状态、锁文件标识、软件／驱动／GPU 信息、配置、精度、误差、延迟分位数、显存和执行状态。
- 统计加速与退化案例、通过率和测量波动；不同 GPU、dtype 或计时协议的结果不得混合汇总。
- 实现固定配置、有限候选搜索、形状规则筛选三种策略。初始规则作为待检验假设；候选数预算不等同于费用控制。
- 结果分析器用明确标注的合成数据验证；合成结果不得进入真实实验汇总或被描述为实测性能。
- 模块验证使用相同权重与输入，对比完整前向输出和耗时，避免将单算子加速等同于模型加速。

## 4. 研究资料与远程交接

中文文档包含：

- 一年研究路线、阶段里程碑及 按阶段安排算力。
- GPU 内存层次、warp／线程块、Triton program、分块与数值精度基础。
- GEMM、在线 softmax、融合 Attention、implicit GEMM 卷积原理。
- Triton IR、LLVM IR、PTX 的阅读路线及编译产物采集方法。
- 文献矩阵：研究问题、方法、已有结论、与本项目关系；明确复现部分和待验证贡献。
- 正确性阈值、计时、消融、配置选择、硬件分析和可复现性协议。
- 本地验收报告、GPU 待验证清单和已知限制。

远程入口准备：

- Colab notebook 只负责环境引导、获取指定代码版本、执行项目命令和导出结果，不维护独立算子实现。
- Linux 指南使用同一源码和锁文件，说明驱动兼容性、输出目录、Nsight 权限和运行顺序。
- 提供 GPU 正确性、Compute Sanitizer、profiling 及编译产物采集入口；缺少工具或权限时明确报告。
- 先做小规模 smoke，再允许完整评测；正确性失败不能继续生成该实现的有效性能结论。
- 每次远程实验记录固定版本与硬件信息，验证产物已保存到持久存储后再结束实例。

账号、订阅和算力余额记录仅在本地保留。

## 5. 执行顺序与验收标准

依次完成：

1. 独立仓库、目录、uv 环境和接口。
2. CPU 参考算法、数学对照与参数检查。
3. Triton 内核草案及 GPU 测试。
4. 实验配置、基线、计时、结果格式和分析工具。
5. 应用模块、远程入口及研究文档。
6. 本地完整验收、修复失败项、形成交付报告。

本地必须实际通过：

- `uv sync --frozen` 成功，包导入和所有 CLI 帮助正常。
- 静态检查与源码语法检查通过。
- CPU 正确性测试通过，包括非整除 GEMM、转置布局、Bias／GELU、softmax 大幅值、在线与显式 Attention 一致性、causal 边界、卷积 padding／stride 和输出尺寸。
- Transformer／CNN 参考路径输出与框架路径一致。
- 误差判据按算子和 dtype 明确配置；除最大绝对误差外报告有稳定分母的相对误差指标，不能靠任意放宽阈值使测试通过。
- 配置解析、错误处理、结果读写、统计和图表输出通过行为测试。
- 完成一次 `doctor → check --device cpu → bench --dry-run → analyze 合成样例` 的本地流程演练。
- 无 CUDA 时自动 GPU 测试明确跳过；显式 CUDA 命令失败退出，且不输出虚假的成功记录。
- 原始运行产物位于 `storage/runs/triton/`，源码仓库不混入环境、数据、日志或凭据。
- 最终报告列出真实通过数量、跳过数量、未验证项、运行命令和产物位置；不得预填测试结果。

GPU 编译和数值验证在本阶段允许保持“待验证”，但代码、测试、配置和执行入口必须完整。**不能以缺少 CUDA 为理由省略可在 Mac 编写或验证的内容，也不能把未运行的 GPU 测试算作通过。**

进入 goal 模式后，以本计划的本地交付和验收标准为完成条件；完成后汇报实际成果与远程待办，不自动扩大为全年 GPU 研究或产生付费操作。
