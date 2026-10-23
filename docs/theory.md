# 算法与硬件学习笔记

## GPU 与 Triton 的映射

全局显存存放大张量；片上共享内存和寄存器容量有限但访问成本较低。相邻线程的地址模式影响访存事务，重复载入数据和中间结果写回都会增加流量。占用率是活跃 warp 与硬件容量的关系，不是越高就一定越快：提高并行度可能压缩单块资源，较大的 tile 也可能提高数据复用却造成寄存器溢出。

Triton 的一个 program 处理张量块，不能简单等同于一个 GPU 线程。`tl.arange` 构造块内索引，`tl.load/store` 的 mask 保护边界，`tl.dot` 表达分块乘加；具体 lane/warp 分配由编译器及目标后端决定。`num_warps`、`num_stages` 与块大小一起影响资源与流水线。当前项目研究参数变化的效果，不直接修改编译器。

学习顺序：向量加法索引与 mask → softmax 归约 → GEMM 地址/复用 → online softmax → Attention 融合 → 卷积虚拟矩阵索引。阅读来源见 sources.md。

## GEMM 与融合

`C[m,n] = Σ_k A[m,k] B[k,n]`，主要计算量约 `2MNK`。一个输出 tile 复用 A 和 B 的载入；增大 tile 可减少重复访存，但会提高寄存器、共享内存与边界浪费。分组调度改变 program 到输出 tile 的映射，尝试增加相邻 program 的数据局部性；收益必须测量。

本项目三种消融：basic（GROUP=1）→ grouped（GROUP=8）→ fused（在累加器中执行 Bias/GELU）。不融合时中间值按 FP32 写出再读取，融合时只写最终低精度结果。GELU 固定为 tanh 近似：

`0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`。

框架 epilogue 基线在 CUDA 使用低精度输入 `torch.mm(..., out_dtype=torch.float32)`，保持 Tensor Core 输入精度与 FP32 epilogue；不能把基线无意改成全 FP32 GEMM 来制造加速。

## 稳定 softmax 与在线归约

稳定形式为 `exp(x-max(x))/sum(exp(x-max(x)))`。处理新块 S 时保留行最大值 m、归一化分母 l 和未归一化输出累加器 a：

```text
m_new = max(m, rowmax(S))
alpha = exp(m - m_new)
p = exp(S - m_new)
l_new = alpha*l + rowsum(p)
a_new = alpha*a + p@V_block
output = a/l
```

这不是近似注意力算法；实数运算下与整体 softmax 相同，但浮点运算顺序和中间精度改变误差。CPU online 参考保留 FP32/FP64，GPU 的 P 为进行 Tensor Core dot 转成输入低精度，因此仍须数值对照。causal 的未来位置设为负无穷并跳过完全无用的未来块，边界载入采用 mask。

## 融合 Attention

标准自注意力为 `softmax(QKᵀ/sqrt(D))V`。显式实现会存下 N×N scores/probabilities。融合实现按 Q 行块流式扫描 K/V，通过在线状态避免完整中间矩阵落显存。Q/K/V/output 本身仍占显存，不能把“不保存 N×N”误解成零额外内存。causal 的计算量统计按有效下三角 pairs，边界 tile 仍可能做额外的被掩码运算。

当前实现是用于研究的基础融合前向，不能称为完整 FlashAttention-2 复刻，更不能默认达到官方实现效率。

## Conv2D 的 implicit GEMM

深度学习 conv2d 使用 cross-correlation，不翻转卷积核。将 `r=(channel,kh,kw)` 作为归约维，输出空间位置加 batch 作为 M，输出通道作为 N：

```text
input_y = output_y*stride - padding + kh
input_x = output_x*stride - padding + kw
Y[b,oc,oy,ox] = Σ_(c,kh,kw) X[b,c,input_y,input_x]*W[oc,c,kh,kw]
```

越界输入贡献零。implicit GEMM 直接计算地址，不生成 im2col 大矩阵；1×1 编译特化后退化为带空间采样的 GEMM。对 NCHW 的 gather 地址开销及非连续访问可能抵消 Tensor Core 收益，这正是迁移验证需分析的限制。

## 编译流程与可验证解释

GPU 上通过 capture 工具收集已编译内核的可用 `asm` 项，通常包括 TTIR、TTGIR、LLVM IR、PTX、cubin；实际键以锁定版本输出为准。先对照索引、load/store 与 dot，再检查目标指令和寄存器/溢出信息。PTX 是中间表示，最终设备机器码仍可能不同；若需要 SASS，使用兼容 CUDA 工具在远程反汇编 cubin。

对每次参数变化记录：候选块大小与 warp/stage → 编译产物 → 寄存器/共享内存 → Nsight 测得带宽/占用情况 → 延迟。只有链条有数据支持时才能写“因何加速”。当前 Mac 只能阅读和准备流程，不能生成目标 GPU 的真实编译证据。
