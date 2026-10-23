# 实验与结果协议

## 正确性先行

所有输入固定种子，框架与自定义实现共用同一份实际输入。低精度参考从已量化的输入提升到 FP32，不重新生成随机数。GEMM/Conv2D 以框架 FP32 运算为 GPU oracle，Attention 以显式 FP32 数学表达式为 oracle；小型模块对比同一参数与低精度边界的框架模块。CPU 测试另外用 FP64 oracle 检查数学实现。

全输出必须有限，形状一致，并逐元素满足 `abs(actual-reference) <= atol + rtol*abs(reference)`。阈值来自 `validation.py`，不得为了让某次实验通过而自动放宽：

| dtype | 默认 atol / rtol | softmax | attention |
|---|---|---|---|
| FP64 | 1e-9 / 1e-8 | 默认 | 默认 |
| FP32 | 2e-5 / 2e-4 | 默认 | 默认 |
| FP16 | 2e-2 / 2e-2 | 2e-3 / 1e-2 | 8e-3 / 2e-2 |
| BF16 | 8e-2 / 5e-2 | 8e-3 / 3e-2 | 4e-2 / 5e-2 |

FP32 Transformer 允许累计误差 1e-4 / 5e-4。阈值是预先声明的初始验收标准，不是误差理论保证；正式研究若需要修订必须记录理由、版本和全部重测结果。额外报告 max_abs、relative_l2（分母最低 1e-12）与 max_scaled_relative（逐元素分母至少为 reference RMS×1e-3 或 1e-12）。后两者是诊断指标，不取代逐元素判据。

## 配置与对照

JSON 中 GEMM shape 顺序为 `[M,N,K]`，Attention 为 `[B,H,N,D]`，Conv2D 为 `[B,C,H,W,OC,kernel]`，Transformer 为 `[B,N,width]`（固定 2 heads），CNN 为 `[B,C,H,W]`。

smoke 覆盖小规模边界与模块；tune 用于规则探索；eval 为不同形状的独立评测集。测试会检查 tune/eval 不重叠（忽略 id 和 dtype）。不能根据 eval 结果修改规则后继续声称它是未见集合；需要新 eval 集。search/pruned 在 eval 上仍会进行在线 autotune，这不代表零调参泛化，必须报告其 tuning_ms。

- torch：框架 eager，是相同语义下的对照；调用 cuBLAS/cuDNN/SDPA 的实际行为由环境决定。
- compile：`torch.compile(fullgraph=True, dynamic=False)`，Inductor 内部 CUDA Graph 关闭。每个 case 创建前重置 Dynamo 进程缓存，不清除磁盘缓存。审计后端记录成功编译的 FX 代码和 targets，要求始终只有一张图；图中断或不支持的算子明确 failed。单图不保证每个 ATen 都被融合为 Triton，仍需检查设备 kernel 名称。
- triton_basic/grouped/fused：只对 GEMM 有意义，保持相同候选起点做调度/融合消融。
- triton_fixed/search/pruned：固定、有限搜索、形状过滤搜索。search 与 pruned 只用于 GEMM/Attention/Conv2D，模块仅支持固定策略。
- sdpa_math/flash/efficient：仅 Attention，强制请求相应 SDPA 后端；不支持或执行失败时保留失败原因，不静默切换到其他后端。

计时之外用 torch.profiler 记录设备 kernel 名称与 SDPA dispatch 事件，供核实实际后端；profiling 不可用时记录原因，不能仅凭标签断言底层库。不要把不适用的后端计入正确性失败率，但不支持的 GPU fused SDPA 必须披露。

## 计时和搜索成本

正式 bench 先运行 GPU smoke 正确性门禁。每个候选先执行并检查精度，通过后才允许测量。若候选全失败，该行不能有有效性能值。

默认 TF32 关闭，FP16/BF16 reduced-precision reduction 关闭，cuDNN benchmark 关闭。默认 `timing.mode=cuda_event`，CUDA events 包围重复调用，采样前预热，事件完成后同步。样本为每组总设备时间除以重复数，再报告 p20/median/p80/min/max。短内核的 event 区间仍可能包含主机提交间隙，不能称为纯指令执行时间。输入传输、oracle 和 correctness 检查不在计时中；算子内部必须的中间分配和转换在计时中。

first_call_ms 是同步的首次调用墙钟耗时，包含当次编译/加载/执行，并不等于纯编译耗时；缓存命中状态可能影响它。tuning_ms 是选择候选的总墙钟时间，包括编译、正确性检查和搜索采样，不包括最终独立计时。记录每个候选的配置、首次调用、检查与搜索中位数。选好配置后重新测量，避免把搜索最优样本直接当报告值。

每个正式硬件至少做 3 次独立运行，并轮换配置中的 backend 顺序观察热状态/负载偏差；现有单次报告不代替跨运行统计。不要混合不同 GPU、dtype、协议或代码版本。调参候选预算不限制平台开机费用。

## 显存和吞吐

独立于计时测量一次调用的 allocator 峰值，记录绝对峰值和相对调用前新增峰值；输出仍存活时读取峰值。它测的是 PyTorch allocator 的 allocated memory，不包括全部驱动/第三方分配，不等同 nvidia-smi 显存。输入、权重及 oracle 的常驻内存影响绝对值，所以比较增量时仍应检查基线。

GEMM 约 2MNK FLOPs；Conv2D 按有效输出乘归约长度；Attention 按两次矩阵乘法、causal 有效 pairs 统计。TFLOPS 不计 softmax/激活标量运算和 mask 后的无用计算，报告中须保留这一定义。模块不输出虚构 FLOPs。

## 产物与失败状态

每个 run 的 metadata.json 保存 schema、kind、完整配置、环境、代码与锁文件标识、输出路径和完成状态；metrics.csv 保存逐 case/backend 记录；preflight.json 保存 GPU smoke 门禁。异常中断必须保留 complete=false。partial 结果可以查看，但不得冒充完整评测。

状态为 passed、incorrect、failed、unsupported。CSV 保存完整采样和候选 JSON 字段，报告显示绝对耗时、波动、配对加速、显存、退化及失败。没有相同 case/shape/dtype 的 torch baseline 就不计算加速。每次 analyze 生成新的子目录，不覆盖先前分析。

只有 kind=gpu_measured 且有 GPU 身份的结果可进入真实报告；synthetic 默认拒绝，显式允许时所有图表仍醒目标记 TEST DATA。CPU correctness JSON 不属于性能结果，不能送入 GPU 性能报告。

## v2 计时与基线审计

`cuda-events-with-wall-v2` 在 event 样本之外记录同步墙钟样本；两者都以 repeats 归一化。预创建 event 句柄，墙钟包含 event 记录与结束同步。保留原 API 调用及校验开销，event 结果依然可能受主机提交间隙影响。

可选 `timing.mode=cuda_graph` 使用 `cuda-graph-replay-v2`：预热后在独立 stream 捕获 repeats 次固定调用，一次 replay 计时。实际 replay 的输出必须再次通过同一 oracle；不支持捕获时明确失败。它消除了重复 Python 调用和分配的开销，但不能用作普通 API 延迟。首尾 event 与 replay 提交开销仍可能影响极短样本，repeats 不能过小。此协议下显存和 kernel names 仍来自捕获之外的普通调用，不是 graph 私有池显存。

CSV 新增 wall_median_ms、wall_samples_ms、timing_mode、compile_audit，metadata 标明 compile_policy。旧 v1 结果保留，不与 v2 混合。`baseline-audit-events.json` 与 `baseline-audit-graphs.json` 使用 tune 的 8 个案例，每种协议各独立运行 3 次、轮换 torch/compile/triton_fixed 顺序。本轮只审计基线与测量方法，未使用 eval 集。PyTorch 2.8 的 `mm(out_dtype=...)` 编译路径若报错，保留 failed 与完整日志，不更换语义或放宽误差来通过。

官方依据：[fullgraph 编程模型](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html)、[CUDA Graph 语义](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)。API 在本项目锁定 PyTorch 2.8 上另行运行验证。

## FP32 输出 GEMM 编译兼容对照

`compile_mm_compat` 是显式命名的另一条编译对照：调用公开注册的 `triton_ops_lab::mm_fp32` 自定义算子，CUDA 实现仍为原来的 `torch.mm(a, b, out_dtype=torch.float32)`，FakeTensor 实现只返回形状、设备、FP32 dtype 与连续布局元信息。它不更改输入精度、FP32 epilogue 或最终低精度舍入边界。此接口为 forward inference，不支持 autograd。

Inductor 将 GEMM 看作不透明外部调用，可以编译周围的 bias/GELU；不能据此声称 Inductor 已修复或直接降低 `aten.mm.dtype`。`compile_audit.mm_policy` 和 `opaque_external_ops` 显式记录这一限制。原 `compile` 后端继续走原生 ATen 路径，失败仍保留，避免静默替换基线。内部算子分派值 `torch_mm_compat` 仅用于这条明确命名的实验路径。

`mm-compat-events.json` 和 `mm-compat-graphs.json` 为 3 个调优形状、FP16/BF16 的融合 GEMM 对照，包含 torch、原生 compile、compile_mm_compat、basic/grouped/fused。每个协议独立做 3 轮，平衡后端先后顺序；不接触 eval。注册正确性由 GPU `torch.library.opcheck`、原生 mm bitwise 比较、融合输出比较、Graph 重放和单图审计共同验证。CPU FakeTensor 检查不能替代这些 GPU 验证。

公开接口依据：[PyTorch custom operator 注册](https://docs.pytorch.org/tutorials/advanced/python_custom_ops_registrations.html)。该兼容方案不修改已安装 PyTorch、不改变 uv.lock。

## GEMM 分块与预绑定对照

`prepare_matmul(a,b,bias,activation=...,config=...)` 复用已校验的设备能力、配置、grid 和常量参数。它绑定具体输入 Tensor 对象，每次调用仍校验 data_ptr、形状、stride、dtype、device、requires_grad，且分配新的输出；允许原地更新输入内容。更换输入对象必须重新准备，修改绑定张量元信息会在 kernel launch 前报错。Graph 重放本身保持标准的静态地址约束。

`triton_prepared` 使用默认 fused tile；`triton_prepared_search` 使用与 `triton_search` 相同的 6 个候选，比较后独立重测。原 `triton_fused`/search/pruned 路径保留全部普通接口检查。分块、warp、stage 选择只在调优集进行，候选表不因 eval 修改；GROUP 固定为 8，本轮不声称搜索了 GROUP。参考 [Triton GEMM 教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)。

新增 construction_ms 记录调用工厂的墙钟时间，first_call_ms 仍从工厂返回后开始；两者不能当作完整冷编译成本。search 的 tuning_ms 含候选构造、首次调用、正确性和选择采样，不含最终独立计时。比较搜索开销回收调用数时，必须说明对照、协议、缓存状态及正的实测延迟差；负收益或不稳定差值不宣称能摊销。

`scripts/gemm_host_profile.py` 另行保存 cProfile、Python 热点与 CPU/CUDA trace。统计含 profiler 开销及可能的驱动等待，不是未插桩主机时间，更不是 kernel latency；诊断不能与正式 bench 样本混用。预绑定是明确的调用契约对照，不得静默替代普通 API 标签。
