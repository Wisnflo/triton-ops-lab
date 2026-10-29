# Triton 算子优化研究

基于 Triton 实现 GEMM、融合 Attention 与受限 Conv2D，研究分块、访存、调度、融合及形状选参，并与 PyTorch、编译基线和融合 SDPA 对照。项目包含 CPU 数学参考、GPU 正确性门禁、独立评测、消融、缓存成本及资源分析工具。

完整说明见 **[项目文档](docs/project.md)**，涵盖实现原理、接口约束、实验协议、性能结果、安全范围、复现命令及参考资料。

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

## 实验覆盖

- 两个独立 L4 会话及一个 A100 会话，各完成 30 个任务、1,284 条记录，覆盖三类算子和 TinyTransformer/TinyCNN。
- 开发调优包含 13 个 GEMM 逐因素配置、384 组消融比较；缓存成本实验包含 42 对冷/热缓存。
- L4 与 A100 均有 28 项 CUDA 检查及 96 项 GPU 正确性测试通过；本地测试为 156 项通过、96 项 GPU 测试按环境要求跳过。

收益取决于形状、精度和计时协议；部分案例相对成熟基线退化。Graph 结果不能代替普通调用延迟。完整应用的厂商库安全告警仍保留，项目范围检查不等于整个框架无告警认证。详见项目文档中的[结果](docs/project.md#5-验证覆盖与结果)和[检查边界](docs/project.md#安全与硬件解释的边界)。

## 目录

| 路径 | 内容 |
|---|---|
| `src/triton_ops_lab/` | 接口、CPU 参考、Triton 内核、模型与实验分析 |
| `configs/`、`tests/` | 实验计划、固定策略和验证 |
| `scripts/`、`notebooks/` | 本地与 GPU 执行入口 |
| `docs/project.md` | 完整项目文档 |
| `tools/` | 发布检查工具 |

所有接口只支持前向推理，明确拒绝未支持的输入与梯度请求，不静默回退。原始测量与完整工具日志保存在实验归档中，不包含在仓库内。
