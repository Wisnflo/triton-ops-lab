# 第三轮：FP32 输出 GEMM 编译兼容与融合消融（2026-10-21）

已在固定环境中验证显式兼容基线 `compile_mm_compat`。源码提交：`4dfe64a689c9fe466762fa51387e057fecd576da`。Notebook：<private-notebook>

## 实现与语义

`compat.py` 通过公开 `torch.library.custom_op` 和 `register_fake` 接口注册 `triton_ops_lab::mm_fp32`。CUDA 实现仍调用原有 `torch.mm(a, b, out_dtype=torch.float32)`；fake 实现仅提供 FP32 输出的形状、设备、连续布局。输入保留 FP16/BF16，FP32 bias/GELU 后再转换为低精度输出；不修改 PyTorch 安装、依赖锁文件、数值阈值或原生 compile 路径。

新基线 `compile_mm_compat` 在单个 FX 图中包含不透明的外部 GEMM，Inductor 编译周围 epilogue。审计明确保存 `mm_policy=explicit_external_fp32_mm`、`opaque_external_ops=[triton_ops_lab.mm_fp32.default]`。这条兼容路径已可用；上游 `aten.mm.dtype` 编译缺陷仍存在，原 `compile` 单独保留失败对照。单图不代表 GEMM 与 epilogue 融合为一个设备 kernel。

此桥接面向 forward inference，不支持梯度。自定义分派可能增加普通调用的主机开销；边界不透明也限制编译器跨 GEMM 的融合。不能将其描述为原生 Inductor 对 `aten.mm.dtype` 的完整支持，也不能把 API 对比的全部差异归因于 GPU 指令优化。

## 验证

Mac：23 项本地验收门禁通过；106 项测试通过，60 项 GPU 测试明确跳过。包含 CUDA FakeTensor、FX/Dynamo 图节点、输出元信息与错误输入检查。首次本地元信息测试误用了真实 CUDA 随机数生成器，已改为 FakeTensor 空张量后重跑通过；首次失败证据保留。

Colab L4：Python 3.12.3、torch 2.8.0、Triton 3.4.0、CUDA wheel 12.8、驱动 580.82.07，与上轮一致。检出精确提交、干净工作树，使用 uv 项目环境。

- 28 项 CUDA 正确性检查通过。
- **60 项 GPU pytest 全部通过**，106 项 CPU 测试未选取。
- 新增 4 项桥接契约测试：两种 dtype、连续与转置输入，桥接与原生 FP32 输出 GEMM 逐位一致，并通过 `torch.library.opcheck` 默认检查。
- 新增 8 项兼容编译测试：两种 dtype，bias+GELU、转置+bias、仅 GELU、TinyTransformer，输出比较、严格单图和 Graph 重放均通过。

## 性能与消融

6 个融合 GEMM 案例，形状 `[M,N,K]` 为 `[65,97,49]`、`[128,128,128]`、`[256,384,192]`，各含 FP16/BF16，均执行 bias+GELU。六个后端：torch、原生 compile、compile_mm_compat、triton_basic、triton_grouped、triton_fused。默认 Triton tile 为 32/64/32，尚未搜索最优配置。

普通 event 与 Graph 协议各 3 次独立 CLI 运行，轮换后端顺序，总计 216 行：**180 passed、36 原生 compile failed**，无 incorrect。每轮 36 行中 30 项通过、6 项原生 compile 失败；全部兼容编译行（36 项）通过单图及外部节点审计。已知失败没有有效性能值，不参与加速比。

Profiler 对兼容基线实际观察到 CUTLASS/cuBLAS GEMM（tail 情况含 cuBLASLt split-K reduce）与 Inductor 的 `triton_poi_fused__to_copy_add_gelu_0`，符合“外部 GEMM + 编译 epilogue”的实现。设备内核名称记录不能代替寄存器、访存计数器或完整 profiler 分析。

以下为固定 Triton fused 的三轮配对加速比中位数；所有比较仅在同一 case/dtype/协议下进行：

| 形状 | dtype | event 对 eager | Graph 对 eager | Graph 对兼容 compile | Graph 对 grouped 未融合 |
|---|---|---:|---:|---:|---:|
| 65/97/49 | FP16 | 1.195x | 3.425x | 1.964x | 1.285x |
| 65/97/49 | BF16 | 1.144x | 3.442x | 1.994x | 1.306x |
| 128/128/128 | FP16 | 1.074x | 2.655x | 1.261x | 1.379x |
| 128/128/128 | BF16 | 1.079x | 2.748x | 1.362x | 1.373x |
| 256/384/192 | FP16 | 1.069x | 2.848x | 1.506x | 1.426x |
| 256/384/192 | BF16 | 1.067x | 3.029x | 1.688x | 1.418x |

Graph 的 fused 对 grouped 比值为 1.285–1.426x，对应这 6 个案例的额外 epilogue 融合收益。Graph 排除重复 Python 调用、校验和动态分配，不能写成普通 API 加速。event 存在主机提交间隙；tail FP16 的某轮对 eager 为 0.858x，另外两轮为 1.195x/1.202x，不能只保留中位数而隐瞒退化。完整跨轮最小/最大值与原始行保留在汇总文件。

这组结果属于调优/开发集和同一 L4 会话，独立 eval 仍未使用。不代表所有 GEMM、大规模模型或不同 GPU 的普遍收益。

## 证据与成本

工作区目录：`storage/runs/triton/20261021T014929Z-colab-mm-compat-6355d8d1/`。

- `triton-mm-compat-evidence.zip` 已下载，ZIP 完整性检查通过。
- `evidence/` 保留 doctor、GPU pytest、6 轮 metrics/metadata/preflight、失败日志、逐轮报告与图表。
- `compat-summary.json` 保存各轮原始配对值、环境和失败原因；`compat-summary.md` 保存分协议中位数及最小/最大值。
- `manifest.json` 保存源码/交付 SHA256，`delivery-status.json` 保存归档 SHA256、GPU 验证状态与释放记录。

账号、订阅和算力余额记录仅在本地保留。

## 下一步

兼容基线现已具备正确性与执行路径证据，下一阶段围绕融合 GEMM 进行 tile/stage/group 调优，量化搜索成本；同时定位普通调用的主机开销。先在调优集冻结策略，再执行保留的 eval。Compute Sanitizer、编译 IR 和硬件计数器分析仍待完成。原生编译缺陷保留最小复现和独立标签，不作为项目内核失败或兼容路径成功的替代证据。
