# 第五轮：GEMM 内存安全、编译产物与硬件资源（2026-10-21）

已采集固定/选中 GEMM 配置的真实编译产物，并成功取得 L4 上的 Nsight 计数器。本轮不重新搜索、不访问保留的 eval，也不生成新的正式性能加速比。

GPU 源码提交：`0e136e09272d0699454e1989c3c6c3506d4b9f8a`，干净检出。Colab notebook（私人地址不随快照提供）。计算内核与第四轮一致，新增的是审计和资源采集工具。文末的退出处理补丁只改变插桩环境与结果分类，不改变算子、数值阈值或性能协议。

## 环境与验收范围

Mac：113 项本地测试通过、68 项 GPU 测试明确跳过；首次交付前 23 项门禁全部通过。退出处理补丁后再次执行同样的本地验收，不能把 Mac 的跳过项算作 GPU 通过。

Colab：NVIDIA L4 / CC 8.9、驱动 580.82.07、Python 3.12.3、PyTorch 2.8.0、Triton 3.4.0、CUDA wheel 12.8。Compute Sanitizer 与 Nsight Compute 均为 2025.3.1.0，使用平台已有工具，未另行安装或更改计数器访问权限。

28 项 CUDA 数值检查通过；正常模式 GPU pytest 为 **68 passed、113 deselected**。正常 GPU pytest 88.81 秒，此数字是整个测试流程耗时，不是算子延迟。

8 个开发案例与第四轮相同：四个 `[M,N,K]` 为 `[65,97,49]`、`[128,128,128]`、`[256,384,192]`、`[768,768,512]`，各含 FP16/BF16，均有 bias+GELU。固定配置为 `(32,64,32,4,2)`；第四轮选中配置对 tail/square 为 `(32,32,32,4,2)`，对 medium/large 为 `(64,64,64,4,3)`，元组依次为 BM/BN/BK/warps/stages。GROUP 固定为 8。

编译采集和 Nsight worker 明确禁用 TF32 与低精度 reduced-precision reduction；worker 自己保存实际环境。父审计进程的 environment 保留框架默认值，不能将父进程字段误当作 worker 的设置。

## 编译产物与资源

8 案例 × 2 配置 × 2 接口，共 **32 组**。每组执行数值检查，保存 source、TTIR、TTGIR、LLVM IR、PTX、cubin、资源元信息和逐文件 SHA256，文件大小和哈希全部核对通过。

同 case、同配置下，公开 fused 与 prepared 的六类产物哈希、寄存器、spill 和编译元信息逐项一致。预绑定改变了主机调用路径，未在这些案例中生成更快的另一套设备代码。这为第四轮“普通调用收益主要来自主机开销减少”的解释提供了额外控制证据。

| M/N/K | dtype | 固定寄存器/线程 | 选中寄存器/线程 | 固定/选中动态共享内存 KiB/CTA |
|---|---|---:|---:|---:|
| 65/97/49 | FP16 | 80 | 56 | 6 / 4 |
| 65/97/49 | BF16 | 80 | 56 | 6 / 4 |
| 128/128/128 | FP16 | 60 | 44 | 6 / 4 |
| 128/128/128 | BF16 | 60 | 44 | 6 / 4 |
| 256/384/192 | FP16 | 60 | 112 | 6 / 32 |
| 256/384/192 | BF16 | 60 | 114 | 6 / 32 |
| 768/768/512 | FP16 | 60 | 104 | 6 / 32 |
| 768/768/512 | BF16 | 60 | 106 | 6 / 32 |

32 组编译对象的原始 `n_spills` 均为 0。这是该编译结果的资源属性，不能推广为所有形状没有 spill、没有内存流量或没有访存瓶颈。

large FP16 的两份 PTX 均为 PTX 8.7 / `sm_89`，可见 `mma.sync...f32.f16.f16.f32`、`ldmatrix`、`cp.async`，两者均未发现 `.local` 声明。静态指令 token 的明细保存在 `ptx-inspection.json`；静态文本次数不是运行时指令计数，不能直接计算吞吐提升。

## Nsight：资源增加与占用率下降的对照

只采集 large FP16、公开 fused 的固定/选中两次 launch；各进行 8 个 profiling replay passes。NVTX 范围只包含被检查的暖调用，oracle 与比较在范围外。`.ncu-rep` 和原始 CSV 均成功导出，worker 记录确认顺序为 fixed、selected，kernel 为 `_gemm`。

| 实际指标 | 固定 | 选中 |
|---|---:|---:|
| Grid thread blocks | 288 | 144 |
| Threads/block | 128 | 128 |
| Registers/thread（报告值） | 60 | 104 |
| Registers/thread（分配粒度后） | 64 | 104 |
| 动态共享内存 bytes/block | 6,144 | 32,768 |
| 驱动共享内存 bytes/block | 1,024 | 1,024 |
| 寄存器约束的 blocks/SM 上限 | 8 | 4 |
| 共享内存约束的 blocks/SM 上限 | 9 | 3 |
| 理论 warp 占用率 | 66.67% | 25.00% |
| 活跃周期 warp 占用率 | 40.02% | 20.01% |
| SM throughput / peak sustained elapsed | 33.65% | 38.38% |

指标名、原始值与单位保存在 `nsight-summary.json`。其中活跃周期占用率为 `sm__warps_active.avg.pct_of_peak_sustained_active`；不能混用 elapsed 周期分母。Nsight 对共享内存配置和驱动保留空间的记录也解释了为什么直接用动态共享内存除以设备最大容量不足以复现占用率上限。

选中分块减少 thread blocks，同时增加每块资源并降低占用率。结合第四轮相同形状 Graph 的 1.235x 配对中位数加速，可确认“提高占用率”不能单独充当选型目标；尚不能把全部收益因果归结为某一种复用或访存机制，因为 tile、BK、stage 同时改变，且 Nsight 仅各一次 launch。

本次 Nsight 插桩中 duration 为 38.400 / 33.376 µs，与第四轮非插桩 Graph 样本不同。它们保留在原始证据中，不作为新的正式性能结果，也不与 Graph 或普通 API 延迟混合计算加速比。未采集完整 MemoryWorkloadAnalysis、SASS 级动态指令/局部访存和独立会话硬件统计，不能据此断言带宽或计算已经饱和。

## memcheck 与主机退出异常

首次以 memcheck 包围全部 GPU pytest：68 passed、113 deselected，pytest 阶段 147.83 秒，Compute Sanitizer 退出码 0、`ERROR SUMMARY: 0 errors`。没有观察到这组执行中的设备内存错误；这不等价于 racecheck/initcheck/synccheck 通过，也不是所有输入内存安全的证明。

完整日志还含 `shutdown_compile_workers` 的 atexit `TimeoutExpired`：Inductor 的 12-worker 子进程池在退出阶段等待 300 秒。最初门禁只检查退出码、测试通过和零错误摘要，误将其作为干净通过；现已修正为 `passed_with_host_shutdown_error`，原始日志和旧 metadata 保留，新的复核判定单独记录，不改写历史证据。

诊断时曾尝试停止该检测进程组，终端返回 `No such process`：原进程已自行退出，该请求未终止检测，也没有触发重跑。之后因完整日志揭示回调异常，才增加一次关闭异步编译的专项复验。

资源审计默认对插桩子进程设置 `TORCHINDUCTOR_COMPILE_THREADS=1` 并记录实际值；显式环境覆盖仍保留。该选择来自锁定 PyTorch 的 `_inductor/config.py` 与 `async_compile.py` 实现，仅改变编译调度，不改变计算内核或正式 bench。

串行编译专项复验使用同一干净 `0e136e0` 源码，通过调用环境设置上述变量，只重跑 memcheck，不重复采集硬件计数器或搜索性能。结果为 **68 passed、113 deselected、零内存错误、退出码 0，且没有 atexit 异常**；pytest 184.97 秒、整个检测进程 189.70 秒。单线程编译的测试阶段较长，但本次消除了额外 300 秒退出等待。这是一次复验，不能推广为所有环境退出问题已解决。后续工具默认设置与异常分类补丁在 Mac 完成 23 项门禁验证；GPU 上验证的是同一原始源码在该环境设置下的行为。

## 证据与成本

[本轮证据目录](data-availability.md)。首轮 ZIP 完整性检查通过，共 2,699 个归档项，SHA256 `df0223c2fc89280341a721ab0b8257a7f1bf88b3d0144e6a50e7b10a08dfa862`。保留完整 Triton cache，原始报告和日志位于 `evidence/20261021T064233Z-gpu-resource-audit-a66e7b69/`；本地派生摘要为 `resource-summary.json/md`、`nsight-summary.json`、`ptx-inspection.json`。

专项复验 ZIP 共 12 项，SHA256 `a552e5bc9c46e65bc8544db5c964fe3902d76d235973fe3d11d699c6ddaa5270`，完整性检查通过，内容在 `serial-evidence/`。`serial-verdict.json` 保存干净通过、环境与指纹；`serial-env.json` 保存实际编译设置；本地 `serial-retry-cell.py` 保留 Colab 复验步骤。

GPU 源码指纹 `d739513a5df8e695eb7cd16d9cb24480e08a64095e207ae69cdc8c252aafb5eb`，锁文件指纹仍为 `d360a200db86e8e1521ef9eaa951f617b155dc69b7afe185d6d8a35208b360f1`。首次本地门禁目录为 `20261021T063355Z-local-acceptance-c3773e62`，退出处理补丁后的门禁目录为 `20261021T065849Z-local-acceptance-b07eaea1`，均在工作区 storage 中。两次本地验收均发生在各自修改提交前，原始 dirty/commit 状态保留。

账号、订阅和算力余额记录仅在本地保留。

下一步冻结开发集上的配置规则及适用契约，在独立 eval 和模块中测泛化，并控制冷/热缓存与搜索摊销。仅根据本轮资源计数器不能宣称研究方法已经优于通用 autotuning，Attention/Conv 的候选调优与更多安全检查也仍未完成。
