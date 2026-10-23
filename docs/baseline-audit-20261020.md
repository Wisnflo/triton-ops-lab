# 第二轮：编译基线与计时审计（2026-10-20）

本轮实现已固定在提交 `0c9b83b32824dcd5c2e20254c0d74812d5389d27`。Notebook：<private-notebook>

## 已完成的修正

- 编译改用 `fullgraph=True, dynamic=False`，每个 case 前重置 Dynamo 进程缓存，避免共享代码对象耗尽重编译上限。Inductor 磁盘缓存仍可复用，首次调用不等于冷编译。
- 自定义审计后端将 FX 图交给锁定版本的 Inductor，保存成功编译的图代码、算子 targets 和图数量。首次调用及计时、profiling 后均要求单图；图中断明确失败。原来的允许图中断基线不再作为完整编译证据。
- 默认 CUDA event 计时新增同步墙钟样本；event 句柄在样本前初始化。
- 新增独立 CUDA Graph 协议：独立 stream 预热、捕获重复调用、重放输出复验，再计时。捕获失败不回退普通调用。
- CSV、metadata、报告显式记录协议、墙钟和编译审计。Graph 模式下的显存及 kernel 名称仍来自普通调用，不表示 graph 私有池内存。
- 修复 Colab 传入 notebook 绘图后端导致 uv 分析进程启动失败的问题，导入 matplotlib 前强制 Agg。
- 新增固定提交的 notebook 生成器，源码只从干净 Git 提交封装，产物输出至外部 storage；锁文件未变。

## 验证及证据

Mac：23 项验收门禁通过；100 项测试通过、48 项 GPU 测试跳过。新增 Dynamo 合约测试实际追踪超过 8 个 case，并验证图中断不能标为编译成功；另做真实 Inductor CPU 编译确认接口。CPU 测试不替代 GPU 结果。

Colab：Linux x86_64、L4（CC 8.9）、驱动 580.82.07，Python 3.12.3、torch 2.8.0、Triton 3.4.0、CUDA wheel 12.8。精确提交、干净工作树，`uv sync --frozen`。

- 28 项 CUDA correctness 检查通过。
- 48 项 GPU pytest 通过，含 6 项新增 Graph 重放测试；100 项 CPU 测试未选取。
- tune 集中的 8 个案例 × 3 后端 × 2 协议 × 3 次独立 CLI 进程，共 144 行；每次轮换后端顺序。
- 每轮 22 项通过、2 项失败，总计 132 passed / 12 failed，无 incorrect。全部 eager 和固定 Triton 路径通过。
- 通过的 compile 每轮 6 行，共 36 行，全部记录单个成功编译图。设备 profiler 显示普通 GEMM 使用 CUTLASS kernel，Attention 使用框架 FlashAttention，卷积包含 cuDNN 与 Inductor layout 转换 kernel；单图并不意味着所有操作融合为一个内核。

本地证据根目录为工作区 `storage/runs/triton/20261020T134817Z-colab-baseline-audit-2b1e74d9/`：

账号、订阅和算力余额记录仅在本地保留。

## 性能观察

下表为固定配置 Triton 相对同输入 PyTorch eager 的配对加速比中位数，方括号为三次运行的最小/最大比值。每轮 event 预热 10、样本 20、重复 20；Graph 预热 10、样本 20、捕获重复 64。不同协议不混算。所有形状来自 tune，不是独立 eval。

| 案例 | 普通调用 event 比值 [min,max] | Graph 重放比值 [min,max] |
|---|---:|---:|
| GEMM 128³ FP16 | 0.381 [0.319,0.400] | 0.907 [0.851,0.963] |
| GEMM 128³ BF16 | 0.379 [0.371,0.381] | 1.038 [1.035,1.039] |
| GEMM bias+GELU M65/N97/K49 FP16 | 1.113 [1.093,1.175] | 3.448 [3.439,3.487] |
| GEMM bias+GELU M65/N97/K49 BF16 | 1.124 [1.100,1.142] | 3.429 [3.422,3.445] |
| causal Attention B1/H2/N512/D64 FP16 | 0.652 [0.636,0.664] | 0.543 [0.543,0.544] |
| causal Attention B1/H2/N512/D64 BF16 | 0.637 [0.578,0.666] | 0.547 [0.547,0.548] |
| Conv B1/C8/H17/W17/OC16/K3/s2/p1 FP16 | 0.424 [0.420,0.430] | 2.234 [2.232,2.235] |
| Conv B1/C8/H17/W17/OC16/K3/s2/p1 BF16 | 0.425 [0.423,0.426] | 2.243 [2.239,2.248] |

Graph 重放排除了重复 Python 分派/校验和动态分配成本，不能将 Graph 数字写作普通 API 延迟或项目整体加速率。event 结果包含主机提交间隙，墙钟包含同步；两者均不能称为纯指令耗时。多轮均在同一 L4 会话完成，平台噪声和不同机器的泛化还需进一步评估。

## 尚未解决的编译限制

含 epilogue 的两种 dtype 在 torch 2.8 的 `torch.mm(..., out_dtype=torch.float32)` fake-tensor 路径稳定报 `meta_mm() takes 2 positional arguments but 3 were given`。每轮明确 failed，graph_count=0，没有有效耗时，不能拿来计算相对 compile 的融合收益。

本轮修复了基线判定和缓存隔离，尚未修复上游编译器的该路径。后续需要独立、可复现地验证兼容软件版本或保持相同数值语义的实现，并更新锁文件后重跑全部对照；不能偷偷改成 FP32 输入 GEMM 后继续使用原基线名称。

## 下一阶段

1. 用最小复现处理上述 out_dtype 编译兼容性，保留旧锁环境和原始失败证据。
2. 以融合 GEMM 为主线，在 tune 集研究调度、分块、epilogue 融合与主机 launch 开销；保留 basic/grouped/fused 消融。
3. 对 Graph 下有收益而普通调用退化的卷积，profiling 分辨 host dispatch、layout 转换和设备执行。Attention 暂作为成熟 FlashAttention 对照及负例，先分析再扩大优化投入。
4. 进行 Compute Sanitizer、编译 IR/资源采集和 Nsight；在调优策略冻结之后再使用尚未触碰的 eval 集。现有结果不能宣称完整研究已完成。

账号、订阅和算力余额记录仅在本地保留。