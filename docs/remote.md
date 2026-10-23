# Linux NVIDIA / Colab 交接

本地交付阶段不执行本文件中的 GPU 命令，不购买或连接算力。远程运行前确认锁定软件与硬件兼容；主机要求 Linux x86_64、NVIDIA GPU CC>=8.0、支持所锁 CUDA wheel 的驱动。驱动由平台管理，`nvidia-smi` 展示的 CUDA 数字不是项目实际 PyTorch 构建版本。不要仅凭“有 GPU”判断兼容。

## 从同一版本开始

先将本地源码审阅并提交到自己的独立仓库，再私有传输或推送到个人 `origin`。本地阶段没有替用户创建远程或发布代码。远程检出精确提交，保持 uv.lock 不变，并确认工作树状态。无远程仓库也可传输包含版本信息的源码包，不传 `.venv`、缓存、凭据。

```sh
# 在远程项目目录，路径替换为实际持久化挂载位置
export TRITON_LAB_OUTPUT_ROOT=/your/persistent/storage/runs/triton
uv sync --frozen
uv run --frozen triton-lab doctor
uv run --frozen triton-lab check --device cuda
uv run --frozen python scripts/gpu_tools.py check
uv run --frozen triton-lab bench --config configs/smoke.json
# 先检查 smoke 输出与成本，再按需要依次执行：
uv run --frozen triton-lab bench --config configs/tune.json
uv run --frozen triton-lab bench --config configs/eval.json
uv run --frozen triton-lab analyze /actual/run/directory
```

显式 CUDA check 失败则停止，不把 pytest 全部 skipped 视为验收。FP16/BF16、非整除、转置、causal、卷积边界和模块输出均需通过；保留 failures 及复现输入。首次完成后再扩大矩阵和序列规模，不直接开始长时间扫参。

PyTorch 的 Linux wheel 将匹配的 CUDA 用户态库和 Triton 一并安装。本项目不需要先编译手写 CUDA 扩展；后续若需要 nvcc、Compute Sanitizer 或 ncu，使用平台提供或 NVIDIA 官方兼容版本。Colab 的预装 torch 可能不同，必须运行 uv 项目解释器，不修改 notebook 全局环境的 torch 来代替锁文件环境。

## 越界、profiling、编译产物

```sh
uv run --frozen python scripts/gpu_tools.py sanitize
uv run --frozen python scripts/gpu_tools.py profile --case gemm-epilogue-float16
uv run --frozen python scripts/gpu_tools.py capture --case gemm-epilogue-float16
```

sanitize 以 Compute Sanitizer memcheck 包围 GPU pytest，返回非零错误码即失败。profile 以 NVTX 范围筛选暖启动自定义执行，保存 ncu 报告与日志；不存在工具时明确失败。性能计数器权限不足常出现 `ERR_NVGPUCTRPERM`，需要平台管理员处理，不能在研究代码里自动修改系统安全配置。

GEMM 专项资源审计：`uv run --frozen python scripts/gpu_resource_audit.py run`。固定使用第四轮 Graph 开发结果的配置表，与默认配置、公开/预绑定接口逐项对照；不读取 eval 或重新选择性能赢家。采集 PTX/cubin 与实际可用 IR，记录寄存器、spill、共享内存及文件指纹。随后以 memcheck 包围 GPU pytest，并尝试仅对 large FP16 的固定/选中公开内核采集 Nsight basic、LaunchStats、Occupancy。所有工具有超时，未安装、权限不足、未生成报告、没有 memcheck 零错误摘要或 pytest 通过记录均不得记为通过。编译资源不等于实际占用率，插桩指标不等于正式性能计时。参考 [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) 和 [Nsight CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)。

通过 `scripts/build_colab_delivery.py --suite gpu-resource` 生成对应的自包含 notebook。该 suite 保留正确性门禁、工具日志和证据导出，不重跑性能搜索。若 Nsight 无计数器权限，保留 `permission_denied`，不修改系统安全设置；memcheck 通过也不等价于 racecheck/initcheck/synccheck 全部通过。

资源审计默认对插桩子进程设置 `TORCHINDUCTOR_COMPILE_THREADS=1`，避免锁定版本在 memcheck 下的异步编译子进程退出等待；显式环境值优先，并记录实际设置。不改变正式 bench 的编译策略。若出现 `Exception ignored in atexit callback`，即使零内存错误和退出码 0，也标为 `passed_with_host_shutdown_error`，不能计作所有工具干净通过。

capture 使用独立 Triton 缓存和可选采集回调，将已编译内核的 asm 与资源元信息保存到本次输出目录。通过 `--config` 和 `--case` 选不同算子。对 basic/grouped/fused 与不同 tile 的深入对照，可在项目 Python 接口传入明确 variant/config，并设置 `TRITON_LAB_CAPTURE_DIR` 为外部存储路径。**正式 bench 拒绝在 capture 环境变量开启时执行**，避免编译产物落盘污染计时。

先阅读 TTIR/TTGIR 的 load/store/dot、再查看 LLVM/PTX 和资源元信息，最后与 ncu 的实测比较；没有 profiling 数据时只能提出瓶颈假设。

## Colab

`notebooks/colab.ipynb` 是薄入口，填入无令牌的私人仓库 URL、完整 commit SHA 和已挂载持久化目录，调用同一套 uv 命令。Notebook 不维护另一份 kernels。Git 凭据用平台 secrets 或正常交互认证，不写入 URL/输出。持久盘挂载由用户完成；本阶段不授予新的云盘访问权限。

账号、订阅和算力余额记录仅在本地保留。

## 保存与释放

确认 metadata、metrics、测试/profiling 日志和分析报告已在持久化目录，检查大小并另行备份到本地 storage 后再停止实例。临时系统盘、Colab `/content` 不视为长期存储。脚本不自动删除实例、不关闭其他任务、不自动购买算力；停止执行单元格或退出 SSH 不等于停止平台计费。

## 基线审计交付入口（v2）

提交并检查干净工作树后执行 `uv run --frozen python scripts/build_colab_delivery.py`，在外部 storage 生成含固定 Git bundle 的 notebook 和 SHA256 manifest。将生成的 notebook 上传至 Colab，选择 L4，依次运行安装、GPU correctness/pytest、双协议三轮审计、分析下载单元格。源码封装采用分行 base64 和隐藏代码，避免 Safari 超长行渲染问题。独立 uv 分析进程强制 Agg 绘图后端。

审计允许 bench 以 1 退出（包含明确失败行），但不忽略 2 等配置/初始化错误；需要完整保留 failed 行。结果下载到本地 storage 并核实 ZIP 完整性后，再释放自己的运行时。生成脚本不自动上传、不买算力、不挂载云盘。

FP32 输出 GEMM 兼容对照使用 `uv run --frozen python scripts/build_colab_delivery.py --suite mm-compat`。这会生成独立 notebook，复用同一套严格门禁、双协议三轮与证据导出流程。原生 compile 的已知失败与 `compile_mm_compat` 的实际结果同时保留。

分块与主机开销对照使用 `--suite gemm-tune`：双协议各三轮，固定/搜索/剪枝和预绑定搜索分别记录；随后对 tail/large FP16 分开运行 cProfile 与 CPU/CUDA trace。候选预算为 6，保持 eval 未使用。所有日志与 trace 仍输出到外部目录，打包核实后再释放运行时。
