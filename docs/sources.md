# 来源、文献矩阵与贡献边界

查阅日期：2026-09-26。项目固定 PyTorch 2.8.0 / Linux Triton 3.4.0；官方 main 文档可能前进，API 细节应与锁定版本源码交叉确认。本地尚未验证 GPU 编译。

| 来源 | 研究问题与已有贡献 | 本项目用途 | 尚需验证 |
|---|---|---|---|
| [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/) | 张量块编程、归约、矩阵乘法和融合示例 | 编程模型和基础实现参考 | 本项目实现的正确性、资源消耗与性能 |
| [矩阵乘法教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) | 分块、分组调度、参数选择 | GEMM 分组 program 映射的衍生实现 | 不同形状下 grouping 与融合是否有收益 |
| [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) | 在线更新 softmax 归一化状态 | online reference 与融合递推的数学基础 | FP16/BF16 中间精度误差 |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | 通过分块降低注意力中间结果的显存 IO | IO-aware attention 前向的算法来源 | 与成熟 SDPA 的显存/延迟对比 |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) | 改进并行与工作划分，减少非矩阵计算和通信 | 分析 program/warp 划分的阅读材料 | 本项目基础内核并未实现全部 FA2 优化 |
| [Triton 融合 Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) | Triton 中实现融合注意力 | 指导 tile、mask、online state 的实现 | 当前草案不能冒充官方教程完整复现 |
| [torch.mm 2.8](https://docs.pytorch.org/docs/2.8/generated/torch.mm.html) | CUDA FP16/BF16 输入可输出 FP32 | 让框架 epilogue 基线保持低精度输入与 FP32 中间输出 | 远程版本及实际内核路径 |
| [torch.nn.attention 2.8](https://docs.pytorch.org/docs/2.8/nn.attention.html) | 显式选择 SDPA 后端 | 数学/Flash/efficient 对照 | 后端是否可用，实际 dispatch 与性能 |
| [uv 与 PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/) | 按平台管理依赖和索引 | Mac/Linux 环境隔离与锁定 | Linux 同锁文件安装及驱动兼容 |
| [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) | CUDA 内存访问等运行期检查 | memcheck 验收入口 | 真实 GPU 上无越界/无未初始化访问 |
| [Nsight Compute 权限说明](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters) | 性能计数器访问限制 | 租用前检查分析权限 | 实际主机 profiling 是否成功 |
| [Triton 3.4 编译器源码](https://github.com/triton-lang/triton/blob/v3.4.0/python/triton/compiler/compiler.py) | CompiledKernel 的 asm 与资源信息 | 可选编译产物采集回调 | 目标后端输出键与资源字段 |

## 哪些属于复现，哪些属于假设

分块 GEMM、分组调度、在线 softmax、融合 Attention 和 implicit GEMM 都是已有方法。本项目代码重写与教学拆解不构成新算法贡献。可能形成贡献的部分是：受资源约束的有限候选筛选及其跨形状效果、融合收益的适用边界、不同算子的统一评测和失败分析；这些必须以强基线、独立评测、调参成本和原始数据支持，当前不作已实现创新的结论。

## 代码来源与许可

GPU 草案按公式和教程思想编写；特别是 GEMM 的分组映射与官方教程具有直接衍生关系，向量索引、softmax 与 attention 编程模式也受官方示例启发。保留 Triton 原始 MIT 许可及版权声明于下方；该声明适用于相关来源/衍生片段，不替第三方 PyTorch、NumPy 等依赖重新授权，也不表示上游作者对本项目性能或正确性背书。论文算法引用不等于复制论文全文。

原始许可：[Triton v3.4.0 LICENSE](https://github.com/triton-lang/triton/blob/v3.4.0/LICENSE)。

```text
Copyright 2018-2020 Philippe Tillet
Copyright 2020-2022 OpenAI

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files
(the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
