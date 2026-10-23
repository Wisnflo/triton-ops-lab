# 首轮 Colab GPU 验证（2026-10-20）

固定源码提交 `5363f830194640036b0da77e664828371c203718` 已完成首次真实 GPU 正确性及 smoke 验证。此记录不代表年度研究、完整调优或性能优势验证完成。

Notebook：<private-notebook>

## 环境与证据

- NVIDIA L4，计算能力 8.9，显存约 22.0 GiB；驱动 580.82.07。
- Python 3.12.3、PyTorch 2.8.0、Triton 3.4.0、CUDA wheel 12.8。
- 从 Git bundle 恢复精确提交，干净工作树；`uv sync --frozen` 创建项目 `.venv`。
- 原始日志、metrics、metadata、正确性结果、分析图表已下载，ZIP 完整性检查通过。
- 工作区证据根目录：`storage/runs/triton/20261020-colab-delivery/`；`evidence/` 为下载包的解压内容。
- CUDA 检查：`evidence/20261020T130822Z-check-cuda-9693d6f5/`。
- GPU pytest：`evidence/20261020T130825Z-check-f049043c/execution.log`。
- 基准：`evidence/20261020T130925Z-smoke-7c947da4/`。
- 报告：该基准目录内 `20261020T131103Z-analysis-742e387b/report.md`。

## 验证结果

28 项 CUDA 正确性检查通过；42 项 GPU pytest 通过，95 项 CPU 测试未选取。覆盖五类算子、FP16/BF16、GEMM 变体、attention 分块与 causal、卷积边界、TinyTransformer/TinyCNN 输出。

Smoke 的 220 个算子/后端组合中，108 项通过、112 项不适用；不适用原因全部为 `backend does not apply to operator`，并非硬件拒绝或测试通过。没有记录 failed/incorrect。候选预算为 2，不代表完整 tune/eval 搜索空间已验证。

## 性能观察及限制

以下均为单次 smoke、固定配置 Triton 相对同输入 PyTorch eager 的中位数，GEMM 配置形状为 M=17、N=23、K=19，含 bias 与 GELU：

| dtype | PyTorch ms | Triton ms | 比值（torch / triton） | 延迟降低 |
|---|---:|---:|---:|---:|
| FP16 | 0.161632 | 0.129323 | 1.250x | 20.0% |
| BF16 | 0.154219 | 0.135083 | 1.142x | 12.4% |

固定配置 Triton 在其余 18 个 smoke case 中均慢于 eager。微小形状、较少样本（warmup=3、samples=5、repeats=3）及主机提交开销使本轮不能支撑典型工作负载加速结论；不得挑选最佳值作为项目整体加速率。需要独立多次复测和代表性 eval 形状。

`compile` 的 20 行均通过输出比较并产生计时，但源码使用 `fullgraph=False`，日志出现 `aten.mm.dtype`/`meta_mm` 错误及 `recompile_limit (8)`。所以“通过”只说明该调用路径输出与计时完成，不能说明全部算子已由 Inductor 编译。正式比较前必须审计图中断、编译覆盖率、缓存隔离及实际设备内核；当前 compile 数字不得作为完整编译基线的优势证据。

## 交付中修正及下一阶段

原 notebook 的超长 base64 单行导致 Safari 无响应，交付版改为分行字符串并隐藏源码单元格。Colab 将 `MPLBACKEND=module://matplotlib_inline.backend_inline` 传入子进程，独立 uv 环境没有该后端，初次分析失败；导出单元格显式设为 `Agg` 后成功，失败日志仍保留。

下一阶段先修复/审计 compile 基线和小尺寸计时方法，再进行 tune/eval。Compute Sanitizer、Nsight、编译 IR 采集及第二 GPU 尚未执行。

账号、订阅和算力余额记录仅在本地保留。