# Mac 本地验收记录

首次验收日期：2026-09-26；技术实验阶段本地复验日期：2026-10-22，该次 23 项门禁通过，148 项测试通过、96 项 GPU 测试在 Mac 明确跳过。2026-10-23 整理分享阶段新增 8 项隐私保护测试，当前本地全量测试为 156 项通过、96 项 GPU 测试跳过。必做 GPU 技术实验的完整验收已通过，最新范围和证据见 [README](../README.md#技术验收与证据)。下方早期记录保留当时状态，不能用其待办状态替代最新验收。

## 2026-10-20 首次提交前复验

在 macOS 27.0.1 arm64 上重新执行 `uv run --frozen python scripts/local_acceptance.py`：

- 95 passed、42 GPU skipped、0 failed；10 个 CPU correctness smoke 用例全部通过。
- 23 项命令验收全部符合预期，包括锁文件安装、静态/语法检查、三套配置 dry-run、合成报告流程和无 CUDA 时的显式拒绝。
- 可执行源码指纹与锁文件 SHA256 均与首次验收一致，未变更算子实现。
- 首次本地 Git 提交包含源码、配置、锁文件、无输出 notebook 和文档；不包含 `.venv`、缓存、凭据或运行日志，不创建远程或推送。
- 复验发生在首次提交之前，因此原始日志中的 `git_commit=null` 与 `git_dirty=true` 保留为当时的真实状态。提交后的版本使用 `git rev-parse HEAD` 获取。

本次原始证据：

- [复验汇总](data-availability.md)
- [pytest 输出](data-availability.md)
- [CPU correctness](data-availability.md)

以下为 2026-09-26 首次验收的历史记录。

## 实际环境与版本

- macOS 27.0 arm64，Apple M4 / 24 GB。
- 项目独立 Python 3.12.14 / `.venv`，PyTorch 2.8.0、NumPy 2.5.3、Matplotlib 3.11.2。
- `uv.lock` 为 Mac 和 Linux x86_64 解析；Linux 依赖中匹配 Triton 3.4.0 / CUDA 12.8 用户态库。Linux 安装和驱动兼容尚未实际验证。
- Mac 环境中未安装 Triton 或 CUDA；doctor 正确报告 local_ready=true、cuda.ready=false。
- 独立 Git 根为 `labs/triton`；父工作区无 `.git`，未配置远程、未推送或发布。首次验收时源码尚未提交，git_commit=null、git_dirty=true，另记录可执行源码指纹。
- 源码指纹：`2990c414557eddfd3b78a941a6f26b02004197b1da1796968afab8a0d9f763e7`。
- 锁文件 SHA256：`d360a200db86e8e1521ef9eaa951f617b155dc69b7afe185d6d8a35208b360f1`。

## 已执行的验收

`pytest`：**95 passed，42 skipped，0 failed**。跳过项全部要求真实 Linux NVIDIA GPU，不能算通过。
CPU CLI correctness：10 个 smoke case 全部通过，包括 Transformer/CNN。完整测试另外覆盖 FP64 oracle、非整除/转置 GEMM、GELU、softmax 极值、online causal Attention、卷积 stride/padding、非法参数、报告统计、合成数据拒绝和正确性门禁。

| 验收项 | 实际退出码 | 预期退出码 | 结果 |
|---|---:|---:|---|
| `sync` | 0 | 0 | 通过 |
| `ruff` | 0 | 0 | 通过 |
| `format` | 0 | 0 | 通过 |
| `syntax` | 0 | 0 | 通过 |
| `pytest` | 0 | 0 | 通过 |
| `help-root` | 0 | 0 | 通过 |
| `help-doctor` | 0 | 0 | 通过 |
| `help-check` | 0 | 0 | 通过 |
| `help-bench` | 0 | 0 | 通过 |
| `help-analyze` | 0 | 0 | 通过 |
| `help-gpu_tools` | 0 | 0 | 通过 |
| `help-synthetic_demo` | 0 | 0 | 通过 |
| `doctor` | 0 | 0 | 通过 |
| `cpu-check` | 0 | 0 | 通过 |
| `dry-run-smoke` | 0 | 0 | 通过 |
| `dry-run-tune` | 0 | 0 | 通过 |
| `dry-run-eval` | 0 | 0 | 通过 |
| `cuda-check-refusal` | 2 | 2 | 通过 |
| `cuda-bench-refusal` | 2 | 2 | 通过 |
| `gpu-tools-refusal` | 2 | 2 | 通过 |
| `synthetic` | 0 | 0 | 通过 |
| `synthetic-refusal` | 2 | 2 | 通过 |
| `analysis` | 0 | 0 | 通过 |

退出码 2 的项目是预期拒绝：Mac 无 CUDA 或分析器禁止把合成数据当真实结果。它们不是 GPU 算子通过的证据。部分工具测试模拟 CUDA 边界以检查调度/错误处理，不验证 GPU 内核或性能。

## 原始证据与输出位置

全部运行产物在工作区 storage 内，不复制到源码仓库：

- [验收汇总](data-availability.md)
- [pytest 原始输出](data-availability.md)
- [JUnit 测试结果](data-availability.md)
- [CPU correctness](data-availability.md)
- [合成数据分析报告](data-availability.md)
- [合成数据图表](data-availability.md)

图表已经人工视觉检查：标签清晰，标题注明 SYNTHETIC / NOT GPU RESULTS。此图只用于报告流程验收，不支持任何性能结论。历史试运行保留在独立目录，最终证据以本记录指向的运行目录为准。

## 计划逐项核对

| 交付要求 | 对应证据 | 本地状态 |
|---|---|---|
| 独立仓库、uv 环境与锁文件 | 根目录、pyproject/uv.lock/.venv，sync 日志 | 已完成 |
| 目录与输出归属 | artifacts 路径解析测试、外部运行目录、git 忽略检查 | 已完成 |
| 五个算子公共接口与输入检查 | ops.py、参数/布局/精度拒绝测试 | 已完成 |
| 三类 CPU 算法及数学验证 | references、FP32/FP64 测试 | 已完成 |
| Triton 实际计算草案 | kernels 中五类内核、GEMM 消融与 mask，静态/语法检查 | 源码完成，GPU 未验证 |
| Transformer/CNN 同权重对照 | models、模块测试及 CPU smoke | CPU 已通过 |
| 三套配置、搜索/筛选与公平基线 | configs、selection、cases、disjoint 测试 | 本地准备完成 |
| 计时、精度门禁、失败记录 | runner、模拟门禁测试、显式无 CUDA 拒绝 | 控制逻辑已测，GPU 计时未测 |
| JSON/CSV、统计与图表 | serialization/analysis 测试和合成分析演练 | 已完成 |
| GPU 检查、memcheck、ncu、编译产物入口 | gpu_tools、capture、帮助和语法检查 | 入口完成，GPU 执行待验证 |
| Colab 同版本薄入口 | notebook 无输出、语法与薄入口测试 | 已完成，未远程运行 |
| 中文研究资料与来源 | plan/roadmap/theory/protocol/remote/sources | 已完成 |

## GPU 待验证与已知限制

- [ ] Linux 使用同一锁文件安装；确认 NVIDIA 驱动、GPU 能力和 BF16 支持。
- [ ] 五类内核及所有候选真实编译；FP16/BF16、causal、边界/转置及卷积参数测试通过。
- [ ] Transformer/CNN 的 GPU 替换路径输出通过。
- [ ] Compute Sanitizer 无越界等错误；Nsight 权限及 NVTX 范围正常。
- [ ] 捕获 TTIR/TTGIR/LLVM/PTX/cubin 中实际提供的产物，核实寄存器/溢出/共享内存。
- [ ] 核实 eager/compile/SDPA 实际设备内核，验证 torch.mm FP32 epilogue 路径。
- [ ] 同型号硬件多次独立计时、显存与调参成本采样；完整披露失败和退化。
- [ ] 独立评测形状及模块级收益支持研究结论，必要时第二种 GPU 验证。

账号、订阅和算力余额记录仅在本地保留。

重跑本地验收：`uv run --frozen python scripts/local_acceptance.py`。此脚本面向当前无 CUDA 的 Mac，不应用来替代远程 GPU 验收。

## 2026-10-20 远程验证更新

后续用户已授权并完成首轮 Colab L4 验证：28 项 CUDA 检查、42 项 GPU 测试通过，smoke 108 项通过、112 项不适用。原始证据已回收，运行时已释放。上述“没有远程连接”的描述仅指早前 Mac 阶段。完整结果与 compile 基线限制见 [Colab 验证记录](colab-validation-20261020.md)。未执行的 sanitizer、profiling、完整候选搜索和独立性能评测仍待完成。

第二轮已修正 fullgraph 基线审计和计时协议，并完成 L4 双协议各 3 轮复测：100 项本地测试、28 项 CUDA 检查、48 项 GPU 测试通过；6 轮共 132 行通过、12 行为明确的 out_dtype 编译失败。详细证据、性能限制及后续主线见 [第二轮审计](baseline-audit-20261020.md)。这不代表上游 out_dtype 编译问题已经修复。

2026-10-21 第三轮完成显式 `compile_mm_compat` 路径并在同锁环境验证：106 项本地测试、28 项 CUDA 检查、60 项 GPU 测试通过；六轮融合 GEMM 对照共 180 行通过、36 行原生 compile 已知失败。全部 36 项兼容编译行通过单图和外部调用审计。记录见 [第三轮兼容与融合验证](mm-compat-validation-20261021.md)。这条兼容路径保留不透明 GEMM 边界，上游原生路径问题仍未修复。

2026-10-21 第四轮完成预绑定 GEMM、有限配置搜索和主机开销分析。23 项本地验收门禁通过，111 项本地测试通过、68 项 GPU 测试在 Mac 明确跳过；Colab 上 28 项 CUDA 检查、68 项 GPU 测试通过。双协议各三轮共 336 行及 1,044 次候选尝试全部通过，两组主机分析完成，证据已归档、L4 已释放。固定配置预绑定的普通调用收益与 Graph 分块收益分开披露，独立 eval 未使用，sanitizer/硬件计数器仍待完成。详见 [第四轮搜索与主机开销验证](gemm-tuning-validation-20261021.md)。

2026-10-21 第五轮完成 32 组 GEMM 编译产物及资源对照，公开/预绑定的同配置产物和资源逐项一致；Nsight 实际采集 large FP16 的固定/选中两次 launch，各 8 个 replay passes。113 项本地测试、28 项 CUDA 检查及正常模式 68 项 GPU 测试通过。首次 memcheck 零设备内存错误，但伴随 Inductor atexit 300 秒超时，已保留并修正门禁分类；关闭异步编译专项复验为 68 项 GPU 测试通过、零内存错误、无退出异常。默认插桩设置与分类补丁通过 23 项本地门禁。两次 L4 已释放，未使用 eval，racecheck/initcheck/synccheck 与更完整 profiling 仍未完成。详见 [第五轮安全与硬件资源验证](gpu-resource-validation-20261021.md)。
