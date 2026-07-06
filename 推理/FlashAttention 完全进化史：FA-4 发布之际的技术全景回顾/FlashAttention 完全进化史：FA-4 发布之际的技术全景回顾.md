
## Attention 的性能瓶颈在哪里

### 一个训练场景

假设我们在训练 [GPT-3](https://zhida.zhihu.com/search?content_id=266453688&content_type=Article&match_order=1&q=GPT-3&zhida_source=entity) 规模的模型，序列长度 2048，hidden dimension 12288，96 个注意力头，那么每个头的 head\_dim 是 128。我们把注意力机制拆开看：

```python
# 标准的 self-attention 实现
def standard_attention(Q, K, V):
    # Q, K, V shape: [batch, seqlen, num_heads, head_dim]
    # 为了简化，这里只看单个头的情况
    # Q, K, V: [seqlen, head_dim] = [2048, 128]
    
    S = Q @ K.T  # [2048, 2048]
    P = softmax(S, dim=-1)  # [2048, 2048]
    O = P @ V  # [2048, 128]
    return O
```

计算量（FLOPs）采用 2 FLOPs/乘加 的口径：

-   `Q @ K.T` 需要 2048 × 2048 × 128 × 2 = 1.07 GFLOPs
-   `P @ V` 需要 2048 × 2048 × 128 × 2 = 1.07 GFLOPs
-   总共约 2.15 GFLOPs（前向，不算 softmax）

但内存访问呢？我们需要（为便于说明，假设 S、P 使用 fp32；Q、K、V、O 使用 fp16）：

-   从 HBM 读取 Q, K, V：3 × 2048 × 128 × 2 bytes = 1.5 MB
-   写入 S 到 HBM：2048 × 2048 × 4 bytes = 16 MB（假设用 fp32）
-   读取 S，写入 P：32 MB
-   读取 P 和 V，写入 O：16 MB + 0.5 MB + 0.5 MB
-   总计约 66 MB 的 HBM 访问（只是前向）

[A100](https://zhida.zhihu.com/search?content_id=266453688&content_type=Article&match_order=1&q=A100&zhida_source=entity) 的算力峰值是 312 TFLOPs/s（FP16 Tensor Core），HBM 带宽约 1.6 TB/s（40GB）/ 2.0 TB/s（80GB）。我们算一下这个 kernel 的算术强度：

```text
算术强度 = FLOPs / 内存访问字节数
        = 2.15 × 10^9 / (66 × 10^6)
        ≈ 32.6 FLOPs/Byte
```

看起来还不错？但问题在于，S 与 P 两个各约 16 MB 的矩阵必须完整地写入和读取 HBM。我们看 [Roofline 模型](https://zhida.zhihu.com/search?content_id=266453688&content_type=Article&match_order=1&q=Roofline+%E6%A8%A1%E5%9E%8B&zhida_source=entity)：

```text
理论计算时间 = 2.15 GFLOPs / 312 TFLOPs/s ≈ 6.9 μs
理论访存时间 = 66 MB / 1.5 TB/s ≈ 44 μs
```

访存时间是计算时间的 13 倍。这个 kernel 是彻底的内存受限（memory-bound）。

### 为什么会产生这么多内存访问

核心问题在于 N×N 的中间矩阵。我们把内存访问按照来源分类：

**必要的访问**（输入输出）：

-   读 Q, K, V：每个 N×d，共 3Nd
-   写 O：N×d
-   总计：4Nd

**额外的访问**（中间结果）：

-   S 矩阵：写入 N²，读取 N²
-   P 矩阵：写入 N²，读取 N²
-   总计：4N²

当序列长度 N 远大于 head\_dim d 时（比如 N=2048, d=128），额外访问 4N² 会远超必要访问 4Nd：

```text
4N² / 4Nd = N/d = 2048/128 = 16 倍
```

这还没算反向传播。标准的反向传播需要保存完整的 P 矩阵，否则无法计算梯度。这意味着训练时每个注意力层要额外占用 N² 的激活内存。

对于 GPT-3（96 层，96 个头，N=2048），仅注意力的激活内存就需要：

```text
96 层 × 96 头 × 2048² × 2 bytes ≈ 74 GB
```

单个样本就要 74GB，这是无法接受的。

### GPU 内存层级的现实约束

我们看一下 A100 的内存层级：

| 层级 | 容量 | 带宽 | 延迟 |
| ----- | ----- | ----- | ----- |
| HBM | 40-80 GB | 1.6 TB/s（40GB）/ 2.0 TB/s（80GB） | ~300 cycles |
| L2 Cache | 40 MB | （数量级）~12 TB/s | ~200 cycles |
| L1+SMEM | 总 192 KB/SM（最大可配置 SMEM ≈ 164 KB） | （数量级）~30 TB/s | ~30 cycles |
| Registers | 64K×32-bit/SM（≈256 KB） | （数量级）~200 TB/s | ~1 cycle |

注意力计算的矛盾在于：

-   我们需要存储 N² 的矩阵，这要求大容量（HBM）
-   但我们的计算密度低，需要高带宽（SMEM/Registers）

能不能不把 S 和 P 写到 HBM，直接在 SMEM 里完成所有计算？

### SMEM 的容量限制

A100 每个 SM 有 192 KB 的 SMEM。假设我们想在 SMEM 里完成一个 block 的计算，需要存储：

```text
# 假设 block size 是 B_r × B_c
Q_block: B_r × d       # 输入 Q 的一块
K_block: B_c × d       # 输入 K 的一块  
V_block: B_c × d       # 输入 V 的一块
S_block: B_r × B_c     # 中间的注意力分数
```

以 fp16 计算，总的 SMEM 需求是：

```text
(B_r × d + B_c × d + B_c × d) × 2 + B_r × B_c × 4 bytes（S 用 fp32）
```

如果我们想处理完整的 N=2048，d=128，需要：

```text
(2048 × 128 + 2048 × 128 + 2048 × 128) × 2 + 2048 × 2048 × 4
= 1.5 MB + 16 MB = 17.5 MB
```

远超 192 KB 的限制。我们必须分块（tiling）。

但这带来一个新问题：softmax 是一个全局操作，它需要知道整行的最大值和总和。如果我们只看到一部分列，怎么计算 softmax？

### 一个可行的 tile 示例（数量级预算）

以 `B_r = B_c = 128, d = 128` 为例，按 fp16 计算（2B/元素），若临时在片保留一块 S（fp32）：

-   Q\_block: 128 × 128 × 2B ≈ 32 KB
-   K\_block: 128 × 128 × 2B ≈ 32 KB
-   V\_block: 128 × 128 × 2B ≈ 32 KB
-   S\_block: 128 × 128 × 4B ≈ 64 KB

合计 ≈ 160 KB，接近 “最大可配置 SMEM ≈ 164 KB/SM” 的上限。实际实现中通常不会让整块 S 长时间驻留 SMEM，而是以寄存器/片上流水的中间态短暂存在，以便预留额外 scratch 与提高并发度。

### Softmax 的全局依赖问题

标准的 softmax 需要两遍扫描：

```python
def softmax(x):  # x shape: [N]
    # 第一遍：找最大值（数值稳定性）
    m = max(x)
    
    # 第二遍：计算 exp 和 sum
    exp_x = exp(x - m)
    s = sum(exp_x)
    
    # 第三遍：归一化
    return exp_x / s
```

如果我们的输入是分块来的，比如 `x = [x1, x2, x3]` 三块：

```python
# 只看到 x1 时
m1 = max(x1)
exp_x1 = exp(x1 - m1)
s1 = sum(exp_x1)
p1 = exp_x1 / s1  # 这个结果是错的！因为 m1 不是全局 max

# 看到 x2 后发现 max(x2) > m1，需要修正 p1
# 怎么修正？
```

这就是 [FlashAttention](https://zhida.zhihu.com/search?content_id=266453688&content_type=Article&match_order=1&q=FlashAttention&zhida_source=entity) 要解决的核心问题：如何在分块计算的情况下，仍然得到正确的 softmax 结果，且不需要回头修改已经处理过的数据。

### 问题总结

到这里，我们面临三个互相冲突的约束：

1.  **计算正确性**：softmax 需要全局信息（整行的 max 和 sum）
2.  **内存容量**：SMEM 只有 192 KB，无法存下 2048×2048 的完整矩阵
3.  **性能要求**：不能把中间的 N² 矩阵写入 HBM，否则内存带宽成为瓶颈

传统的做法是放弃约束 3，老老实实把 S 和 P 写入 HBM。FlashAttention 的突破在于找到了一种算法，能够在满足约束 1 和 2 的同时，也满足约束 3。

## 在线 Softmax 的数学推导

在分块处理时，我们希望得到与全量 softmax 完全一致的结果。关键是对每一行维护两个统计量：当前最大值 `m_i` 和未归一化的指数和 `l_i`，然后逐块更新累积输出。

具体来说，处理第 j 个列块时，假设 `S_ij = Q_i K_j^T`：

首先计算本块的统计量：

```text
m_ij = rowmax(S_ij)
P_tilde_ij = exp(S_ij - m_ij)
l_ij = rowsum(P_tilde_ij)
```

然后与历史统计量对齐：

```text
m_new = max(m_i, m_ij)
α = exp(m_i - m_new)
β = exp(m_ij - m_new)
l_new = α · l_i + β · l_ij
```

最后更新累积输出：

```text
P_ij = P_tilde_ij · (β / l_new)
O_i = O_i · (α · l_i / l_new) + P_ij · V_j
```

更新统计量 `m_i = m_new, l_i = l_new`，继续处理下一个列块。

处理完所有列块后，`O_i` 就是最终输出。前向传播只需写回 `O` 和每行的 LSE（log-sum-exp，可表示为 `m + log(l)`），反向传播时用这些信息重构所需的局部量，完全避免了把 N×N 注意力矩阵写入 HBM。

## 在线 Softmax 的数学突破

### 分块 Softmax 的朴素尝试

我们先来看一个简化的场景。假设一行注意力分数被分成两块：

```text
S = [S1, S2]  # S1: [B_c] 第一块，S2: [B_c] 第二块
```

朴素的想法是：先对 S1 做 softmax，再对 S2 做 softmax，最后合并。但这显然是错的：

```python
# 错误的做法
P1 = softmax(S1)  # [0.2, 0.3, 0.5]
P2 = softmax(S2)  # [0.1, 0.4, 0.5]
P = concat([P1, P2])  # [0.2, 0.3, 0.5, 0.1, 0.4, 0.5]
sum(P) = 2.0  # 不等于 1！
```

正确的 softmax 应该是：

```python
P_correct = softmax([S1, S2])
# 需要用全局的 max 和 sum
```

问题的关键在于：当我们只看到 S1 时，计算出的局部 max 和 sum 不是最终的全局值。那能不能在看到 S2 后，用某种方式修正之前的结果，而不需要重新计算？

### 在线更新的数学推导

我们从 softmax 的定义出发：

```python
# 对于完整的输入 x = [x1, x2, ..., xn]
m = max(x)
p_i = exp(x_i - m) / sum_j(exp(x_j - m))
```

现在把 x 分成两块 `[x(1), x(2)]`，我们来推导如何从局部统计量得到全局结果。

**第一步：处理 x(1)**

```python
m_1 = max(x(1))
d_1 = sum(exp(x(1) - m_1))  # 归一化因子
```

此时我们得到的是局部 softmax：

```text
p(1) = exp(x(1) - m_1) / d_1
```

**第二步：看到 x(2) 后的更新**

新的全局最大值是：

```text
m_new = max(m_1, max(x(2)))
```

这里有两种情况：

**情况 A**：如果 `max(x(2)) <= m_1`，那么 `m_new = m_1`，全局 max 没变。此时：

```text
# x(1) 部分不需要修正（max 没变）
p(1)_new = exp(x(1) - m_1) / d_new

# x(2) 部分需要用 m_1 计算
p(2)_new = exp(x(2) - m_1) / d_new

# 其中 d_new = d_1 + sum(exp(x(2) - m_1))
```

**情况 B**：如果 `max(x(2)) > m_1`，那么 `m_new = max(x(2))`，全局 max 改变了。此时：

```text
# x(1) 部分需要修正（因为 max 变了）
p(1)_new = exp(x(1) - m_new) / d_new
         = exp(x(1) - m_1 - (m_new - m_1)) / d_new
         = exp(x(1) - m_1) * exp(-(m_new - m_1)) / d_new
         = p(1)_old * d_1 * exp(m_1 - m_new) / d_new
```

注意最后一步：`exp(x(1) - m_1) = p(1)_old * d_1`，这是我们已经计算过的。

统一两种情况，我们定义修正因子：

```text
alpha = exp(m_1 - m_new)  # 如果 m_new = m_1，则 alpha = 1
```

那么更新公式是：

```text
# 旧块的修正
p(1)_new = (p(1)_old * d_1 * alpha) / d_new

# 新块的计算
m_2 = max(x(2))
d_2 = sum(exp(x(2) - m_2))
beta = exp(m_2 - m_new)
p(2)_new = (exp(x(2) - m_2) * beta) / d_new

# 新的归一化因子
d_new = d_1 * alpha + d_2 * beta
```

### 从 Softmax 到 Attention Output 的累积

我们真正需要的不是 P 本身，而是 `O = P @ V`。如果我们能直接累积输出，就不需要存储完整的 P 矩阵。

假设 V 也按列分块：`V = [V(1); V(2)]`（按行拼接）。我们有：

```text
O = P @ V = [P(1), P(2)] @ [V(1); V(2)]
  = P(1) @ V(1) + P(2) @ V(2)
```

在只看到第一块时：

```text
O_1 = (exp(x(1) - m_1) / d_1) @ V(1)
```

看到第二块后：

```text
O_new = P(1)_new @ V(1) + P(2)_new @ V(2)
      = (p(1)_old * d_1 * alpha / d_new) @ V(1) + (d_2 * beta / d_new) * softmax(x(2)) @ V(2)
      = (alpha * d_1 / d_new) * O_1 + (beta * d_2 / d_new) * O_2
```

其中 `O_2 = softmax(x(2)) @ V(2)` 是第二块的局部输出。

这个公式告诉我们：我们可以用加权的方式累积输出，权重由修正因子 alpha、beta 和归一化因子 d 决定。

### FlashAttention-1 的完整前向算法

现在我们把上面的数学翻译成实际的算法。这是 FA-1 论文中的 Algorithm 1：

```python
def flash_attention_forward(Q, K, V, B_r, B_c):
    """
    Q: [N, d] 查询矩阵
    K: [N, d] 键矩阵  
    V: [N, d] 值矩阵
    B_r: Q 的块大小（行方向）
    B_c: K, V 的块大小（列方向）
    """
    N, d = Q.shape
    T_r = ceil(N / B_r)  # Q 的块数
    T_c = ceil(N / B_c)  # K, V 的块数
    
    # 初始化输出（在 HBM 中）
    O = zeros([N, d], dtype=float32)
    l = zeros([N], dtype=float32)  # 归一化因子
    m = full([N], -inf, dtype=float32)  # 行最大值
    
    # 外层循环：遍历 Q 的块
    for i in range(T_r):
        # 加载 Q 的一块到 SMEM
        Q_i = Q[i*B_r : (i+1)*B_r, :]  # [B_r, d]
        
        # 为当前 Q 块初始化累积器（SMEM/寄存器）
        O_i = zeros([B_r, d], dtype=float32)
        l_i = zeros([B_r], dtype=float32)
        m_i = full([B_r], -inf, dtype=float32)
        
        # 内层循环：遍历 K, V 的块
        for j in range(T_c):
            # 加载 K, V 的一块到 SMEM
            K_j = K[j*B_c : (j+1)*B_c, :]  # [B_c, d]
            V_j = V[j*B_c : (j+1)*B_c, :]  # [B_c, d]
            
            # 计算当前块的注意力分数（在 SMEM 中）
            S_ij = Q_i @ K_j.T / sqrt(d)  # [B_r, B_c]
            
            # 在线 softmax 更新
            m_ij = rowmax(S_ij)  # [B_r]，当前块的行最大值
            P_ij = exp(S_ij - m_ij[:, None])  # [B_r, B_c]
            l_ij = rowsum(P_ij)  # [B_r]，当前块的行和
            
            # 计算新的全局统计量
            m_new = maximum(m_i, m_ij)  # [B_r]
            
            # 修正因子
            alpha = exp(m_i - m_new)  # 旧块的修正
            beta = exp(m_ij - m_new)   # 新块的修正
            
            # 更新归一化因子
            l_new = alpha * l_i + beta * l_ij  # [B_r]
            
            # 更新输出（关键步骤）
            O_i = (alpha * l_i / l_new)[:, None] * O_i + (beta * l_ij / l_new)[:, None] * (P_ij @ V_j)
            
            # 更新统计量
            m_i = m_new
            l_i = l_new
        
        # 将当前 Q 块的结果写回 HBM
        O[i*B_r : (i+1)*B_r, :] = O_i
        l[i*B_r : (i+1)*B_r] = l_i
        m[i*B_r : (i+1)*B_r] = m_i
    
    # 返回输出和统计量（统计量用于反向传播）
    return O, l, m
```

### 实现中的几个关键点

**SMEM 的使用规划**

在每次内层循环中，片上资源（SMEM/寄存器）需要存储：

```text
Q_i: B_r × d × 2 bytes (fp16)
K_j: B_c × d × 2 bytes (fp16)
V_j: B_c × d × 2 bytes (fp16)
S_ij: B_r × B_c × 4 bytes (fp32，提高数值稳定性)
O_i: B_r × d × 4 bytes（若放在寄存器更佳，累积需要更高精度）
m_i, l_i: B_r × 2 × 4 bytes (fp32)
```

以 A100 的 L1+SMEM 总 192 KB/SM（最大可配 SMEM ≈ 164 KB）为例，选择 `B_r = B_c = 128, d = 128`：

```text
总需求（上界估算）≈ (128×128 + 128×128 + 128×128) × 2B  # Q, K, V（fp16）
                 + 128×128 × 4B                 # S（fp32，短暂驻留）
                 + 128×128 × 4B                 # O（若放寄存器则不占 SMEM）
                 + 128×2 × 4B                   # m, l（fp32）
≈ 98 KB + 64 KB + 64 KB + 1 KB = 227 KB（若 O 保存在寄存器、S 短驻可降低 SMEM 占用）
```

实际实现通常将累加器 O 放在寄存器，S 以“计算—消费—丢弃”的方式短暂存在，从而把 SMEM 占用控制在可配上限内；必要时再调小 tile（如 64–128）。

**为什么 O\_i 用 fp32**

这是数值稳定性的关键。我们在累积输出：

```text
O_i = alpha * O_i + beta * (P_ij @ V_j)
```

如果用 fp16，经过多次累加（T\_c 次，可能是几十次），累积误差会很大。论文中测试了 fp16 和 fp32 的精度差异，fp32 可以保证最终结果与标准实现的误差在 1e-3 量级。

**因果掩码的处理**

对于 GPT 等自回归模型，需要因果掩码（causal mask），即位置 i 不能看到位置 j > i。在代码中：

```python
# 在计算 S_ij 之后
if causal:
    # 构造掩码：对于 Q 的第 i 块，K 的第 j 块
    # 如果 i*B_r + local_i < j*B_c + local_j，则设为 -inf
    mask = create_causal_mask(i, j, B_r, B_c)
    S_ij = S_ij + mask  # 加 -inf 会在 exp 后变成 0
```

### IO 复杂度分析

我们来算一下这个算法的 HBM 访问量。

**外层循环**（Q 的块）：执行 `T_r = N / B_r` 次

**内层循环**（K, V 的块）：每次外层循环执行 `T_c = N / B_c` 次

**每次内层循环的 HBM 访问（按标量计）**：

-   加载 Q\_i：B\_r × d（外层循环只加载一次；分摊到内层为 B\_r × d / T\_c）
-   加载 K\_j：B\_c × d
-   加载 V\_j：B\_c × d
-   合计：B\_r × d / T\_c + 2 × B\_c × d

**总的 HBM 访问**：

```text
总访问（前向，按标量计）= T_r × T_c × (B_r × d / T_c + 2 × B_c × d)
                      = T_r × (B_r × d + 2 × T_c × B_c × d)
                      = (N / B_r) × (B_r × d + 2 × (N / B_c) × B_c × d)
                      ≈ N × d + 2 × N² × d / B_r

# 还需计入：
# 1) 写回 O：+ N × d
# 2) 写回 LSE（或等效行统计）：+ O(N)

总访问（前向，齐次口径）≈ N × d + 2 × N² × d / B_r + N × d + O(N)
```

其中前两项为 Q/K/V 相关访存的主项，后两项分别为 O 写回与 LSE 写回，口径与“标准实现”对齐。

由此可见，当 B\_r 固定（例如 64–128）且 N 远大于 B\_r 时，主项近似为 `≈ 2 × N² × d / B_r`。

对比标准实现（需要读写 S 和 P 矩阵，各 N² 量级的读/写）：

```text
标准实现 = 4 × N × d  # Q, K, V, O
         + 4 × N²    # S, P 各读写两次
```

当 N ≫ B\_r 且 d 为常见头维（64/128）时，FlashAttention 的主项随 `N²` 增长但前系数显著更小（与 `B_r` 成反比），因此相较标准实现的 `≈ 4×N²` 读写量获得显著下降（数量级 2× 及以上，具体取决于 tile 与实现细节）。

但这还不是最优的。论文中证明了，在 SMEM 大小为 M 的情况下，最优的块大小选择可以让 HBM 访问降低到：

```text
O(N² × d² / M)
```

对于 A100（M = 192 KB），这比标准实现减少了约 7-9 倍的 HBM 访问。

### [Triton](https://zhida.zhihu.com/search?content_id=266453688&content_type=Article&match_order=1&q=Triton&zhida_source=entity) 实现的核心循环

我们来看 FA-1 在 Triton 中的实际实现（来自 `flash_attn_triton_og.py`）：

```python
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, L, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 当前线程块处理的 Q 块
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Q 块的索引
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # 加载 Q 块
    q_ptrs = Q + off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :]
    q = tl.load(q_ptrs)
    
    # 初始化累积器
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # 遍历 K, V 块
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # 加载 K 块
        k_ptrs = K + off_hz * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :]
        k = tl.load(k_ptrs)
        
        # 计算 QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale
        
        # 在线 softmax
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        # 更新统计量
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        l_new = alpha * l_i + beta * l_ij
        
        # 更新输出
        acc_scale = alpha * l_i / l_new
        acc = acc * acc_scale[:, None]
        
        # 加载 V 块并累积
        v_ptrs = V + off_hz * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :]
        v = tl.load(v_ptrs)
        
        p_scale = beta / l_new
        acc += tl.dot(p * p_scale[:, None], v)
        
        # 保存新的统计量
        m_i = m_new
        l_i = l_new
    
    # 写回结果
    off_o = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :]
    tl.store(Out + off_o, acc)
    tl.store(L + off_hz * N_CTX + offs_m, m_i + tl.log(l_i))
    tl.store(M + off_hz * N_CTX + offs_m, m_i)
```

这段代码直接对应了我们前面的算法。注意最后存储的 L 是 `m_i + log(l_i)`，这就是 log-sum-exp (LSE)，用于反向传播。

## FlashAttention-2 的三个关键优化

### FA-1 还有哪些性能瓶颈

FA-1 已经通过 IO 优化大幅提升了性能，但在 A100 等设备上，内核的硬件利用率仍显不足（Tensor Core 利用率常低于 30%）。问题集中在：

```text
指标                     实际值        理论峰值      利用率
SM 占用率               52.3%         100%         52.3%
Tensor Core 利用率      28.7%         100%         28.7%
SMEM 带宽              8.2 TB/s      ~30 TB/s     27.3%
寄存器使用             128/255                    50.2%
Warp 执行效率          68.4%         100%         68.4%
```

几个瓶颈：

1.  SM 占用率偏低：并行度不够；
2.  Tensor Core 利用率偏低：非 matmul 操作占比过大；
3.  SMEM 带宽未打满：数据搬运与同步仍有冗余。

### 优化一：减少非 matmul FLOPs

FA-1 的内层循环中存在大量逐元素与缩放操作，A100 上 FP32 标量 FLOPs 吞吐约为 FP16 Tensor Core 的 1/16：

```text
FP16 Tensor Core: 312 TFLOPs/s
FP32 CUDA Core:   19.5 TFLOPs/s
比值: 16:1
```

FA-2 的核心做法是延迟归一化，将“每次循环 2 次缩放”改为“末尾一次性归一”，把时间更多地花在 matmul 上：

```python
# 初始化
O_i = zeros([B_r, d])
l_i = zeros([B_r])

for j in range(T_c):
    # ... 计算 m_ij, l_ij, P_ij ...
    O_i = alpha[:, None] * O_i + beta[:, None] * (P_ij @ V_j)
    l_i = alpha * l_i + beta * l_ij

# 循环结束后：一次性归一
O_i = O_i / l_i[:, None]
```

节省的标量操作（示例，T\_c=16）：

```text
FA-1: T_c × 2 × B_r × d
FA-2: 1 × B_r × d
节省比例 ≈ (2T_c−1)/(2T_c) ≈ 96.9%
```

### 优化二：增加并行度

FA-2 在前向保持行块并行的同时，在反向将“外层循环”切换为 Q 块，使得 Q 块之间完全独立，GPU 调度器更易填满 SM 并隐藏延迟：

```python
# FA-1 反向（外层 K/V）
for j in range(T_c):     # 串行
    for i in range(T_r): # 可并行
        atomic_add(dQ[i], ...)

# FA-2 反向（外层 Q）
for i in range(T_r):     # 可并行
    for j in range(T_c): # 串行
        atomic_add(dK[j], ...)
        atomic_add(dV[j], ...)
```

实测（示例，A100，seqlen=2048）：

```text
FA-1 反向: SM 占用率 ~52%, 耗时 188 ms
FA-2 反向: SM 占用率 ~78%, 耗时 134 ms
提升: ~1.4x
```

### 优化三：改进 Warp 内的工作划分

FA-1 使用 Split‑K（按列切 K/V），软最大与行和需要跨 warp 归约，产生 SMEM 往返与同步；FA-2 改为 Split‑Q（按行切 Q），每个 warp 独立完成完整行的 softmax 与累加，避免跨 warp 通信。

```text
# FA-1: 每个 warp 算 S 的一部分列，需要 __syncthreads() 合并行统计
# FA-2: 每个 warp 负责若干完整行，K/V 广播到所有 warp，softmax 行级归约在 warp 内完成
```

性能对比（示例，依实现而变）：

| 指标 | FA-1 (Split-K) | FA-2 (Split-Q) | 提升 |
| ----- | ----- | ----- | ----- |
| SMEM 事务数 | 8 次/iteration | 2 次/iteration | 4x |
| Warp 同步次数 | 2 次/iteration | 0 次/iteration | — |
| 寄存器使用（占用） | 96⁄255 | 128⁄255 | — |

实测吞吐（A100，seqlen=2048，d=64）：

```text
FA-1: 104 TFLOPs/s
FA-2: 171 TFLOPs/s
提升: ~1.64x
```

### 三个优化的综合效果

FA-2 论文（A100，前向+反向）报告：

| 序列长度 | FA-1 | FA-2 | 加速比 |
| ----- | ----- | ----- | ----- |
| 512 | 28 ms | 19 ms | 1.47x |
| 1024 | 85 ms | 51 ms | 1.67x |
| 2048 | 312 ms | 156 ms | 2.0x |
| 4096 | 1180 ms | 580 ms | 2.03x |

硬件利用率：

```text
FA-1: ~25–30% 峰值性能
FA-2: ~50–55% 峰值性能
```

FA-2 将注意力核效率推近 GEMM：

```text
cuBLAS GEMM (A100, FP16): ~280 TFLOPs/s（~90% 峰值）
FA-2 Attention: ~171 TFLOPs/s（~55% 峰值）
```

考虑到 attention 含大量非 matmul 操作（softmax、逐元素），该效率已较高。

## FlashAttention-3：Hopper 架构的异步突破

### FA-2 的遗留瓶颈

FA-2 在 A100 上已经达到了 ~55% 的峰值利用率，这对于包含大量非 matmul 操作的 attention kernel 来说已经很不错了。但如果我们深入分析 kernel 的执行时间线，会发现一个无法回避的问题：

```text
# FA-2 在 A100 上的典型时间分布（单个 CTA，单次内层循环）
总时间: 100 μs
├─ 数据搬运（GMEM → SMEM）: 25 μs  (25%)
├─ 计算 QK^T: 30 μs                (30%)
├─ Softmax: 15 μs                   (15%)
├─ 计算 PV: 25 μs                   (25%)
└─ 同步与杂项: 5 μs                 (5%)
```

问题在于这些操作是**串行**的。所有 warp 必须等待数据搬运完成才能开始计算，计算完成后才能搬运下一批数据。即使计算单元空闲，也无法提前开始工作。

我们能否让数据搬运和计算**同时进行**？这需要两个条件：

1.  **硬件异步能力**：数据搬运不占用计算线程
2.  **软件解耦机制**：生产者（搬运）和消费者（计算）独立工作

Hopper 架构（H100）提供了这两个能力。

### Hopper 的三个关键硬件特性

**特性 1：TMA (Tensor Memory Accelerator)**

传统的 GMEM → SMEM 数据搬运需要线程参与：

```python
// A100 上的数据搬运（FA-2）
__shared__ float smem_K[128][128];

// 所有线程协作搬运
for (int i = threadIdx.x; i < 128*128; i += blockDim.x) {
    int row = i / 128;
    int col = i % 128;
    smem_K[row][col] = gmem_K[row][col];  // 每个线程搬一部分
}
__syncthreads();  // 必须等待所有线程完成
```

H100 的 TMA 是**硬件 DMA 引擎**，发起后不占用计算线程（示意，具体 API 以 CUDA/TMA/CUTLASS 版本为准）：

```python
// H100 上的 TMA 搬运（FA-3）
__shared__ float smem_K[128][128];

// 只需一个线程发起，硬件自动完成
if (threadIdx.x == 0) {
    cute::copy_async(
        tma_desc,           // TMA 描述符
        gmem_K,             // 源地址
        smem_K              // 目标地址
    );
}
// 无需全体 __syncthreads()，但需配合命名屏障/等待语义保证可见性
```

性能对比（示例，搬运 128×128 fp16 矩阵；数值依形状/环境而变）：

```text
A100 手动搬运: ~8 μs，占用所有 warp
H100 TMA 搬运: ~6 μs，仅需少量线程发起，不占用 compute warps
```

**特性 2：WGMMA (Warp Group Matrix Multiply-Accumulate)**

A100 的 MMA 指令是单 warp（32 线程）级别的，H100 引入了 warp group 级别的 WGMMA，4 个 warp（128 线程）协同完成更大的 tile：

```python
// A100: 单 warp MMA
// 每个 warp 独立计算 16×8 的输出 tile
mma.sync.aligned.m16n8k16.f32.f16.f16.f32 {out}, {A}, {B}, {C};

// H100: Warp group WGMMA  
// 4 个 warp 协同计算 64×64 的输出 tile（示例尺寸）
wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16.f32 {out}, {A_desc}, {B_desc}, {C};  // 示意语法
```

好处：

-   更大的 tile → 更高的数据复用
-   更少的指令开销
-   更好的调度灵活性

**特性 3：异步屏障与命名屏障**

A100 只有全局的 `__syncthreads()`，所有 warp 必须同步。H100 支持**命名屏障**，允许不同组的 warp 独立同步：

```python
// H100 的命名屏障
__shared__ cuda::barrier<cuda::thread_scope_block> barrier_load;
__shared__ cuda::barrier<cuda::thread_scope_block> barrier_compute;

// Producer warp 只在 barrier_load 上同步
if (warp_role == Producer) {
    load_data();
    barrier_load.arrive_and_wait();
}

// Consumer warp 只在 barrier_compute 上同步
if (warp_role == Consumer) {
    barrier_load.wait();  // 等待数据就绪
    compute();
    barrier_compute.arrive_and_wait();
}
```

这使得生产者-消费者模式成为可能。

### FA-3 的异步流水线设计

FA-3 将 CTA 内的 warp 分为两组角色：

**角色分配（示例，总共 12 warps/CTA）**

```python
// Producer warps（1-2 个）
// 职责：用 TMA 从 HBM 加载 K, V 到 SMEM
if (warp_id == 0) {
    role = Producer;
}

// Consumer warps（10-11 个）
// 职责：从 SMEM 读取数据，执行 WGMMA 和 softmax
if (warp_id >= 1 && warp_id <= 11) {
    role = Consumer;
}
```

**2-Stage 流水线**

关键是维护两个 stage 的 SMEM buffer，让加载和计算重叠：

```python
// SMEM 布局
__shared__ float K_smem[2][128][128];  // 2 个 stage
__shared__ float V_smem[2][128][128];

// 流水线主循环
for (int kv_idx = 0; kv_idx < num_kv_tiles; ++kv_idx) {
    int stage_load = (kv_idx + 1) % 2;  // 下一个要加载的 stage
    int stage_comp = kv_idx % 2;        // 当前要计算的 stage
    
    // Producer: 加载 stage_load
    if (role == Producer) {
        // 等待 stage_load 的 buffer 空闲（上一轮计算已完成）
        barrier_empty[stage_load].wait();
        
        // TMA 加载（异步，不阻塞）
        tma_load(K_smem[stage_load], K_gmem[kv_idx + 1]);
        tma_load(V_smem[stage_load], V_gmem[kv_idx + 1]);
        
        // 通知 stage_load 已就绪
        barrier_full[stage_load].arrive();
    }
    
    // Consumer: 计算 stage_comp
    if (role == Consumer) {
        // 等待 stage_comp 的数据就绪
        barrier_full[stage_comp].wait();
        
        // WGMMA 计算（使用 stage_comp 的数据）
        S = wgmma(Q, K_smem[stage_comp]);
        P = softmax(S);
        O_acc += wgmma(P, V_smem[stage_comp]);
        
        // 通知 stage_comp 的 buffer 已空闲
        barrier_empty[stage_comp].arrive();
    }
}
```

**时间线对比**

FA-2（同步）：

```text
时间 →
├─ Load K₀  (所有 warp 参与)
├─ Sync
├─ Compute K₀ (所有 warp 参与)
├─ Load K₁
├─ Sync  
├─ Compute K₁
└─ ...
```

FA-3（异步 2-stage）：

```text
时间 →
├─ Load K₀  (Producer warp)
│   └─ Compute K₀ (Consumer warps，等待 K₀)
│
├─ Load K₁  (Producer warp)
│   └─ Compute K₁ (Consumer warps) ← 与 Load K₂ 重叠
│
├─ Load K₂  
│   └─ Compute K₂ ← 与 Load K₃ 重叠
└─ ...
```

时间节省（示例）：

```text
FA-2 单次迭代: Load(25μs) + Compute(60μs) = 85μs
FA-3 单次迭代: max(Load(25μs), Compute(60μs)) = 60μs
理论加速: 85/60 = 1.42x
```

### GEMM 与 Softmax 的两阶段交错

FA-3 进一步优化了 softmax 的调度。传统做法是串行：

```python
// 传统顺序（FA-2）
S = Q @ K.T      // GEMM 1
P = softmax(S)   // Softmax（慢，标量密集）
O = P @ V        // GEMM 2
```

问题：softmax 是标量操作，无法用 Tensor Core，会让 Tensor Core 空闲。

FA-3 的优化是将一个 Q tile 对应的多个 K/V tile 的计算**交错**进行：

```python
// FA-3 的交错调度（伪代码，简化）
for kv_idx in range(0, num_kv_tiles, 2):  # 每次处理 2 个 K/V tile
    # 阶段 1：两个 GEMM（QK）可以连续调度 WGMMA
    S0 = Q @ K[kv_idx].T      # WGMMA 计算 S0
    S1 = Q @ K[kv_idx+1].T    # WGMMA 计算 S1（与 S0 流水）
    
    # 阶段 2：交错执行 softmax 和 PV
    # Warp group A 做 softmax(S0)，Warp group B 同时做 P1 @ V（如果有）
    P0 = softmax(S0)          # 标量操作
    O += P0 @ V[kv_idx]       # WGMMA 计算 PV
    
    P1 = softmax(S1)  
    O += P1 @ V[kv_idx+1]
```

通过这种交错，Tensor Core 和标量单元可以并行工作，提高整体吞吐。

### FP8 量化路径

FA-3 是第一个支持 FP8（E4M3 格式）的 FlashAttention 版本。挑战在于 attention 的动态范围很大，直接量化会损失精度（下述范围与上限为通用描述，具体取决于实现口径/框架对 FP8 的定义与舍入规则）。

**FP8 E4M3 的动态范围**

```text
E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits
最小正常值约 2^-6，上限约 448（E4M3 最大有限值）
动态范围: 约 6–7 个数量级（视实现/舍入而略有差异）
```

**Attention 的动态范围**

```python
# Attention 分数
S = Q @ K.T / sqrt(d)  # Q, K 归一化后，S ∈ [-10, 10]

# Softmax 后
P = exp(S) / sum(exp(S))  # P ∈ [near 0, near 1]

# exp(S) 的范围
exp(-10) = 4.5e-5
exp(10) = 22026
# 跨越 9 个数量级！
```

**FA-3 的解决方案：Block Quantization + Incoherent Processing**

核心思想是分块量化，每个块有独立的 scale：

```python
// 分块量化 S（示例，每 16×16 一个块）
constexpr int BLOCK_SIZE = 16;

for (int bi = 0; bi < 128; bi += BLOCK_SIZE) {
    for (int bj = 0; bj < 128; bj += BLOCK_SIZE) {
        // 找到块内最大绝对值
        float block_max = 0;
        for (int i = bi; i < bi + BLOCK_SIZE; ++i) {
            for (int j = bj; j < bj + BLOCK_SIZE; ++j) {
                block_max = max(block_max, abs(S[i][j]));
            }
        }
        
        // 计算 scale（映射到 FP8 的范围）
        float scale = block_max / 448.0f;  // 448 ≈ FP8 E4M3 的最大值
        
        // 量化
        for (int i = bi; i < bi + BLOCK_SIZE; ++i) {
            for (int j = bj; j < bj + BLOCK_SIZE; ++j) {
                S_fp8[i][j] = (fp8)(S[i][j] / scale);
            }
        }
        
        // 保存 scale（用于反量化）
        scales[bi/BLOCK_SIZE][bj/BLOCK_SIZE] = scale;
    }
}
```

**Incoherent Processing（非相干处理）**

关键优化：softmax 的 max 和 sum 可以在 FP8 上近似计算，然后用 FP32 修正：

```python
// 行最大值（FP8 近似）
float m_approx = fp8_rowmax(S_fp8) * scale;  // 快，但不精确

// 指数和（FP8 近似）
float l_approx = fp8_rowsum(exp(S_fp8 - m_approx)) * scale;

// FP32 修正（只需重算几个关键值）
float m_exact = fp32_rowmax(S_critical);  // 只算几个可能的最大值
float correction = exp(m_approx - m_exact);
float l_exact = l_approx * correction;
```

精度实验（论文数据，GPT-3 规模）：

```text
FP16 baseline: Perplexity 10.23
FP8 FA-3:      Perplexity 10.26  (Δ +0.03, 可接受)
FP8 naive:     Perplexity 12.87  (Δ +2.64, 不可接受)
```

### FA-3 的性能表现

论文在 H100 SXM 上的实测数据（seqlen=2048，d=128，前向+反向；以下为论文示例数据，具体数值依实现与环境而异）：

**FP16 吞吐量**

| Kernel | A100 (FA-2) | H100 (FA-2移植) | H100 (FA-3) | FA-3 提升 |
| ----- | ----- | ----- | ----- | ----- |
| 前向 | 171 TFLOPs/s | 245 TFLOPs/s | 740 TFLOPs/s | 3.0x |
| 反向 | 135 TFLOPs/s | 198 TFLOPs/s | 660 TFLOPs/s | 3.3x |
| 总体 | 153 TFLOPs/s | 221 TFLOPs/s | 700 TFLOPs/s | 3.2x |

硬件利用率：

```text
H100 FP16 峰值: 989 TFLOPs/s
FA-3 实测: 700 TFLOPs/s
利用率: 70.8%
```

这已经接近了包含大量非 matmul 操作的 kernel 的理论上限。

**FP8 吞吐量**

| Kernel | H100 FA-3 (FP16) | H100 FA-3 (FP8) | FP8 加速 |
| ----- | ----- | ----- | ----- |
| 前向 | 740 TFLOPs/s | 1250 TFLOPs/s | 1.69x |
| 反向 | 660 TFLOPs/s | 1150 TFLOPs/s | 1.74x |
| 总体 | 700 TFLOPs/s | 1200 TFLOPs/s | 1.71x |

```text
H100 FP8 峰值: 1979 TFLOPs/s
FA-3 实测: 1200 TFLOPs/s
利用率: 60.6%
```

FP8 的利用率略低，是因为 block quantization 和 scale 处理带来的额外开销。

**端到端训练加速**

在 GPT-3 175B 模型上的实测（H100 8卡，序列长度 2048）：

```text
配置                    吞吐量 (tokens/s/GPU)    相对 FA-2
FA-2 (A100)             2,340                    1.0x (baseline)
FA-2 (H100, 移植)       3,450                    1.47x
FA-3 (H100, FP16)       5,120                    2.19x
FA-3 (H100, FP8)        6,890                    2.94x
```

### FA-3 的 API 使用

FA-3 通过 `hopper` 子模块暴露：

```python
from flash_attn.flash_attn_interface import flash_attn_func
from flash_attn.flash_attn_triton import flash_attn_func as flash_attn_triton

# 标准 FP16 路径（自动选择最优实现）
# H100 上会自动使用 FA-3
output = flash_attn_func(
    q, k, v,
    causal=True,
    softmax_scale=1.0/math.sqrt(d)
)

# 显式使用 Hopper FP8 路径
import torch
from flash_attn.flash_attn_interface import flash_attn_func

q_fp8 = q.to(torch.float8_e4m3fn)
k_fp8 = k.to(torch.float8_e4m3fn)
v_fp8 = v.to(torch.float8_e4m3fn)

output = flash_attn_func(
    q_fp8, k_fp8, v_fp8,
    causal=True,
    softmax_scale=1.0/math.sqrt(d)
)
```

**与 FA-2 的兼容性**

FA-3 保持了与 FA-2 相同的 Python 接口，但内部实现完全不同：

```text
# FA-2 的实现（A100/Ampere）
- 使用手动加载（线程协作）
- 单 warp MMA 指令
- 全局 __syncthreads()
- 只支持 FP16/BF16

# FA-3 的实现（H100/Hopper）
- 使用 TMA 异步加载
- WGMMA 指令
- 命名屏障
- 支持 FP8
```

运行时自动检测 GPU 架构并选择最优实现：

```python
# flash_attn 内部的调度逻辑（简化）
def flash_attn_func(q, k, v, **kwargs):
    device_arch = torch.cuda.get_device_capability()
    
    if device_arch >= (9, 0):  # Hopper (sm90+)
        return flash_attn_hopper(q, k, v, **kwargs)
    elif device_arch >= (8, 0):  # Ampere (sm80+)
        return flash_attn_ampere(q, k, v, **kwargs)
    else:
        return flash_attn_triton(q, k, v, **kwargs)
```

### FA-3 的工程挑战与取舍

**挑战 1：复杂的状态机**

异步流水线需要精确控制每个 warp 的状态转换，代码复杂度显著增加：

```python
// FA-2: 简单的循环（~200 行核心代码）
for (int j = 0; j < num_kv_tiles; ++j) {
    load_kv(j);
    compute(j);
}

// FA-3: 复杂的状态机（~800 行核心代码）
// 需要管理：
// - 2 个 stage 的生命周期
// - Producer/Consumer 的同步点
// - TMA 完成事件
// - 边界条件（第一次迭代、最后一次迭代）
```

**挑战 2：调试困难**

异步代码的调试非常困难，因为：

-   Producer 和 Consumer 并行执行，时序不确定
-   错误可能在几百个周期后才暴露
-   传统的 printf 调试会破坏时序

开发团队使用了专门的工具：

-   Nsight Compute 的 warp state 分析
-   硬件性能计数器
-   单元测试覆盖边界情况

**取舍：只支持 Hopper**

FA-3 的 Hopper 专用实现不适用于 A100（相关指令/特性在 A100 上不可用），因为：

-   TMA 指令不存在
-   WGMMA 指令不存在  
    
-   命名屏障不存在

这是一个有意的取舍：不兼容旧硬件，换取最大化新硬件性能。FA-2 仍然是 A100 的最优选择。

### 从 FA-2 到 FA-3 的设计演进

如果我们回顾 FA-1 → FA-2 → FA-3 的演进路径：

**FA-1**：算法突破

-   核心贡献：在线 softmax，避免 O(N²) 内存
-   硬件假设：通用 GPU（有 SMEM 即可）
-   实现复杂度：中等

**FA-2**：架构适配

-   核心贡献：减少非 matmul FLOPs，改进并行度
-   硬件假设：Ampere Tensor Core，理解 warp 调度
-   实现复杂度：中等

**FA-3**：深度协同

-   核心贡献：异步流水线，完全重叠 IO 和计算
-   硬件假设：Hopper 特定指令（TMA、WGMMA、屏障）
-   实现复杂度：高

演进的趋势是：**越来越深入硬件细节，换取越来越高的性能**。这也意味着可移植性降低，但对于追求极致性能的场景（如大模型训练），这是值得的。

在 FA-3 的基础上，FA-4 将继续在 Blackwell 上探索更深的异步流水线和更多的 warp 角色分工。但无论硬件如何演进，FA-1 提出的在线 softmax 算法始终是核心。

## FlashAttention-4 的深度异步与 CuTe 编程

### Blackwell 的硬件演进

FA-3 在 H100 上报告的峰值利用率可达 ~75%（论文报告）。Blackwell (B100/B200) 进一步增强了异步能力：

**TMEM (Tensor Memory)**

Hopper 的数据路径：

```text
HBM → TMA → SMEM → WGMMA → Registers
```

Blackwell 引入 TMEM 作为 UMMA 路径的片上缓冲，多个阶段可复用（不同 SKU 的支持情况可能存在差异）。整体数据流要点：

-   HBM 通过 TMA 进入片上（SMEM）；
-   UMMA 从片上（SMEM/TMEM）取数，并将中间/累加结果写入 TMEM 或寄存器；
-   Softmax/归一化阶段从 TMEM 读到寄存器计算；
-   最终通过 TMA 写回 HBM。

**更细粒度的 Warp 专门化（示例）**

FA-3 有 2 种角色（Producer/Consumer），FA-4 扩展到 5 种：

```python
# FA-4 的 warp 角色分配（示例：16 warps/CTA，形状不同会调整）
Load_warp = 1       # Warp 0: TMA 加载
MMA_warp = 1        # Warp 1: UMMA 矩阵乘
Softmax_warps = 8   # Warp 2-9: Softmax 计算
Correction_warps = 4  # Warp 10-13: 在线更新的修正
Epilogue_warps = 2  # Warp 14-15: 写回 HBM
```

每个角色有专门的命名屏障，形成更深的流水线。

### CuTe DSL 的编程模型

FA-4 使用 CUTLASS 的 CuTe (CUDA Templates) DSL，用 C++ 模板描述硬件操作。

**传统 CUDA 写法**（手动管理索引）：

```python
// 加载一个 tile
__global__ void load_tile(float* dst, const float* src, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < M && col < N) {
        dst[row * N + col] = src[row * N + col];
    }
}
```

**CuTe 写法**（描述式）：

```python
// 定义 tensor 的 shape 和 stride
auto tensor_layout = make_layout(
    make_shape(Int<128>{}, Int<64>{}),  // 128 rows, 64 cols
    make_stride(Int<64>{}, Int<1>{})    // row-major
);

auto gmem_tensor = make_tensor(gmem_ptr, tensor_layout);
auto smem_tensor = make_tensor(smem_ptr, tensor_layout);

// Copy：编译器自动生成最优的加载代码
cute::copy(gmem_tensor, smem_tensor);
```

CuTe 的优势：

1.  **编译期优化**：shape/stride 是模板参数，编译器可以完全展开循环
2.  **硬件映射**：自动选择最优的访问模式（向量化、bank conflict 避免）
3.  **可读性**：逻辑和实现分离

FA-4 的核心 kernel 入口（从 `flash_fwd_sm100.py`）：

```python
# Python 接口定义 kernel 的模板参数
@dataclass
class FlashFwdSm100:
    elem_type: str = "cutlass::bfloat16_t"
    head_dim: int = 128
    mma_m: int = 128  # Q tile 大小
    mma_n: int = 128  # K/V tile 大小
    num_stages: int = 2
    num_warps: int = 16
    
    def get_cute_kernel(self):
        # 生成 C++ 模板实例化
        return f"""
        using Kernel = flash::FlashFwdKernel<
            {self.elem_type},
            Shape<Int<{self.mma_m}>, Int<{self.mma_n}>>,
            {self.num_stages},
            {self.num_warps}
        >;
        """
```

### 一个 Tile 的生命周期（概念示例）

FA-4 的关键创新是每个 CTA 同时处理 **2 个 Q tile**：

```text
# 每个 CTA 处理的数据
Q_tile_0: [128, head_dim]  # 第一个 Q tile
Q_tile_1: [128, head_dim]  # 第二个 Q tile
K_tile: [128, head_dim]    # 当前的 K tile
V_tile: [128, head_dim]    # 当前的 V tile
```

这样做的好处是提高 K/V 的复用率，因为两个 Q tile 共享同一份 K/V。

**阶段 1：Load Warp 加载数据**

```python
// Load warp (warp 0)
if (warp_id == 0) {
    // 加载 Q tiles 到 SMEM（只加载一次）
    tma_load(sQ[0], mQ[q_tile_0_offset]);
    tma_load(sQ[1], mQ[q_tile_1_offset]);
    
    // 循环加载 K, V tiles
    for (int kv_idx = 0; kv_idx < num_kv_tiles; ++kv_idx) {
        int stage = kv_idx % kStages;
        
        // 等待 stage 可用
        cute::wait_barrier(kBarrierLoad, stage);
        
        // TMA 加载
        tma_load(sK[stage], mK[kv_idx]);
        tma_load(sV[stage], mV[kv_idx]);
        
        // 通知 MMA warp
        cute::arrive_barrier(kBarrierMMA, stage);
    }
}
```

**阶段 2：MMA Warp 计算 QK^T**

```python
// MMA warp (warp 1)
if (warp_id == 1) {
    for (int kv_idx = 0; kv_idx < num_kv_tiles; ++kv_idx) {
        int stage = kv_idx % kStages;
        
        // 等待 Load warp 完成
        cute::wait_barrier(kBarrierMMA, stage);
        
        // UMMA 计算（直接写入 TMEM）
        // 两个 Q tile 同时计算
        tS0 = umma(sQ[0], sK[stage], tS0_tmem);  // Q0 @ K^T
        tS1 = umma(sQ[1], sK[stage], tS1_tmem);  // Q1 @ K^T
        
        // 通知 Softmax warps
        cute::arrive_barrier(kBarrierSoftmax, stage);
    }
}
```

注意：UMMA 在部分路径可直接将中间/累加结果写入 TMEM，减少寄存器压力；具体取决于内核与形状。

**阶段 3：Softmax Warps 计算概率**

```python
// Softmax warps (warp 2-9, 8个 warps)
if (warp_id >= 2 && warp_id <= 9) {
    int local_warp = warp_id - 2;  // 0-7
    
    for (int kv_idx = 0; kv_idx < num_kv_tiles; ++kv_idx) {
        int stage = kv_idx % kStages;
        
        cute::wait_barrier(kBarrierSoftmax, stage);
        
        // 从 TMEM 读取到寄存器
        auto S_reg = tmem_to_register(tS0_tmem, local_warp);
        
        // 数值稳定：行级减最大值 + 底数转换（e^x = 2^(x/ln2)）
        float row_max = reduce_max(S_reg);
        const float inv_ln2 = 1.4426950408889634f;
        auto P_reg = fast_exp2_approx((S_reg - row_max) * inv_ln2);
        float row_sum = reduce_sum(P_reg);
        
        // 写回 TMEM
        register_to_tmem(tP_tmem, P_reg, local_warp);
        
        // 保存统计量到 SMEM
        smem_stats[local_warp] = {row_max, row_sum};
        
        cute::arrive_barrier(kBarrierCorrection, stage);
    }
}
```

**阶段 4：Correction Warps 做在线更新**

```python
// Correction warps (warp 10-13, 4个 warps; 阈值为工程启发式)
if (warp_id >= 10 && warp_id <= 13) {
    for (int kv_idx = 0; kv_idx < num_kv_tiles; ++kv_idx) {
        int stage = kv_idx % kStages;
        
        cute::wait_barrier(kBarrierCorrection, stage);
        
        // 读取统计量
        auto [m_new, l_new] = smem_stats[...];
        auto [m_old, l_old] = prev_stats;
        
        // 判断是否需要修正（优化：只在 max 显著变化时修正）
        if (abs(m_new - m_old) > threshold) {
            // 从 TMEM 读取累积的输出
            auto O_reg = tmem_to_register(tO_tmem);
            
            // 修正
            float alpha = exp2f(m_old - m_new);
            O_reg *= alpha;
            
            // 写回
            register_to_tmem(tO_tmem, O_reg);
        }
        
        // 更新统计量
        prev_stats = {m_new, l_new};
        
        // 通知可以计算 P @ V（减少不必要修正）
        cute::arrive_barrier(kBarrierOutput, stage);
    }
}
```

这里有个优化：只在 row\_max 显著变化时才做修正。如果新的 K tile 没有更大的值，alpha ≈ 1，不需要修正旧的输出。

**阶段 5：MMA Warp 计算 PV 并累积**

```python
// MMA warp 继续工作（P 在 TMEM，V 在 SMEM）
if (warp_id == 1) {
    for (int kv_idx = 0; kv_idx < num_kv_tiles; ++kv_idx) {
        int stage = kv_idx % kStages;
        
        cute::wait_barrier(kBarrierOutput, stage);
        
        // UMMA 计算 P @ V（P 在 TMEM，V 在 SMEM）
        tO0_tmem = umma(tP_tmem, sV[stage], tO0_tmem);  // 累加到 TMEM
        tO1_tmem = umma(tP_tmem, sV[stage], tO1_tmem);
        
        // 通知可以释放 K/V 的 SMEM
        cute::arrive_barrier(kBarrierLoad, stage);
    }
}
```

**阶段 6：Epilogue Warps 写回结果**

```python
// Epilogue warps (warp 14-15)
if (warp_id >= 14 && warp_id <= 15) {
    // 所有 K/V tiles 处理完后
    cute::wait_all_barriers();
    
    // 从 TMEM 读取最终输出
    auto O0_reg = tmem_to_register(tO0_tmem);
    auto O1_reg = tmem_to_register(tO1_tmem);
    
    // 最终归一化（l_final 来自行级 LSE/统计的累积，分别对应两个 Q tiles）
    O0_reg /= l_final[0];
    O1_reg /= l_final[1];
    
    // 写入 SMEM
    register_to_smem(sO[0], O0_reg);
    register_to_smem(sO[1], O1_reg);
    
    // TMA 写回 HBM
    tma_store(mO[q_tile_0_offset], sO[0]);
    tma_store(mO[q_tile_1_offset], sO[1]);
}
```

### 快速 Exp2 的数学技巧（示例）

FA-4 使用多项式近似替代硬件的 exp 函数：

```python
// 传统方法：调用 SFU (Special Function Unit)
float p = __expf(s - m);  // 慢，吞吐 ~4 TFLOPs/s

// FA-4 方法：多项式近似
float fast_exp2_approx(float x) {
    // 分解：2^x = 2^floor(x) × 2^frac(x)
    int xi = __float2int_rd(x);  // floor(x)
    float xf = x - (float)xi;     // frac(x)
    
    // 整数部分：位移操作（极快）
    int int_part = (xi + 127) << 23;  // 2^xi 的 IEEE 754 表示
    float scale = __int_as_float(int_part);
    
    // 小数部分：三次多项式
    // 在 [0,1] 区间拟合 2^x
    const float c0 = 1.0f;
    const float c1 = 0.693147f;
    const float c2 = 0.240227f;
    const float c3 = 0.0558687f;
    
    float frac_part = c0 + xf * (c1 + xf * (c2 + xf * c3));
    
    return scale * frac_part;
}
```

这个实现使用少量 FMA 指令，常带来核内 softmax 2–20× 的吞吐提升（随形状/流水/数据分布而变）。

精度（示例）：

```python
# 测试 10000 个随机数 [-10, 10]
max_relative_error = 0.0023  # 0.23%
mean_relative_error = 0.0008  # 0.08%

# 对于 bfloat16 (7 bits 精度 ≈ 1%)
# 这个误差在可接受范围内
```

### FA-4 的当前限制

以下为当前已知限制，可能随版本演进而变化。

FA-4 还有一些限制：

**1\. Paged KV 的 page\_size 固定为 128**

```text
# FA-3: 支持任意 page_size 的 Paged KV（Python 接口常传 page_table）
# FA-4: 目前 page_size 固定为 128（tile/页大小为编译期常量）
```

这是因为 CuTe 的模板实例化需要编译期常量。支持可变 page\_size 需要大量的模板组合，会显著增加编译时间。

**2\. head\_dim 组合受限（当前状态）**

```text
# FA-3: 支持 head_dim <= 256，8 的倍数
# FA-4: 只支持特定组合

已支持：64/96/128；前向存在 (qk=192, v=128) 等特定组合；后向目前要求 `d_v == d_qk`。
更多组合（192、256 等）在 TODO/受限。
```

**3\. 暂无 FP8 路径**

```python
# FA-3: 支持 FP8 量化
flash_attn_func(..., dtype=torch.float8_e4m3fn)

# FA-4: 只支持 FP16/BF16
# FP8 的 UMMA 指令映射还在开发中
```

### FA-4 的性能预期（定性）

虽然官方尚未公开完整基准（Blackwell 硬件尚未大规模发布），结合内部建模与文档描述：

```text
主要来源：
- 更深的流水线与角色分工；
- TMEM 减少片上往返；
- 快速 exp2 与近似策略；
- 硬件代际改进。
```

## 跨芯片迁移的抽象与实践

### FlashAttention 的不变量

我们回顾 FA-1 到 FA-4 的演进，提取出几个在所有版本中都保持不变的核心设计：

**1\. 在线 Softmax 算法**

所有版本都使用相同的数学公式来处理分块 softmax：

```python
# 统计量更新
m_new = max(m_old, m_current)
alpha = exp(m_old - m_new)
beta = exp(m_current - m_new)
l_new = alpha * l_old + beta * l_current

# 输出更新
O_new = alpha * O_old + beta * O_current
```

这个算法与硬件无关，只要硬件支持 exp 和基本算术运算就能实现。

**2\. 分块计算（Tiling）**

所有版本都将 Q, K, V 分成小块，在片上内存完成计算：

```python
for q_tile in Q_tiles:
    for kv_tile in KV_tiles:
        # 片上计算
        S_tile = matmul(q_tile, kv_tile)
        P_tile = softmax(S_tile)
        O_tile += matmul(P_tile, V_tile)
```

块大小由片上内存容量决定，但分块策略是通用的。

**3\. 反向重计算**

所有版本都只保存 O(N) 的 LSE 统计量，反向时重计算 P：

```python
# 前向保存
save(O, LSE)  # 总共 N×(d+1)

# 反向重建
S = Q @ K.T
P = exp(S - LSE)  # 重计算
```

这是内存-计算权衡的核心，与具体硬件无关。

### 硬件抽象的三个层次

我们可以把 FlashAttention 的硬件依赖抽象成三层：

**层次 1：存储层级映射**

```python
class MemoryHierarchy:
    # 所有 GPU/NPU/TPU 都有的层级
    global_memory: str    # HBM / DDR / HBM2e
    l2_cache: str         # 硬件管理，透明
    on_chip_fast: str     # SMEM / L1 / SRAM
    registers: str        # 每线程私有
    
    # 容量和带宽
    fast_memory_size: int      # 决定块大小
    global_bandwidth: float    # 决定 IO 优化收益
    fast_bandwidth: float      # 决定计算密度要求
```

FA 的核心优化就是最大化数据在 `on_chip_fast` 中的复用，减少 `global_memory` 访问。

**层次 2：计算单元映射**

```python
class ComputeUnits:
    # 矩阵乘法加速器
    tensor_core_type: str     # Tensor Core / MXU / Systolic Array
    tensor_throughput: float  # TFLOPs/s
    
    # 标量/向量单元
    scalar_throughput: float  # 用于 softmax
    vector_width: int         # SIMD 宽度
    
    # 特殊函数单元
    has_sfu: bool            # 是否有硬件 exp/log
    sfu_throughput: float    # 如果没有，用近似
```

FA-2 的优化（减少非 matmul FLOPs）在任何 `tensor_throughput >> scalar_throughput` 的硬件上都适用。

**层次 3：并发控制映射**

```python
class ConcurrencyModel:
    # 线程层级
    warp_size: int           # 32 (NVIDIA) / 64 (AMD) / wave (其他)
    warps_per_block: int     # CTA 大小
    
    # 同步机制
    has_async_copy: bool     # TMA / CDMA / DMA
    has_named_barrier: bool  # 生产者-消费者模式
    barrier_cost: float      # 同步开销
    
    # 调度
    max_blocks_per_sm: int   # 占用率
    scheduler_policy: str    # 影响流水线设计
```

FA-3/FA-4 的异步流水线需要 `has_async_copy=True` 和 `has_named_barrier=True`。不满足的硬件需要退化到 FA-2 的同步模式。

## 展望一下未来

### FlashAttention 演进的三条主线

回顾 FA-1 到 FA-4 的四代演进，我们能看到三条清晰的优化主线，每条都在不断深化。

**主线 1：IO 效率的持续提升**

这是最核心的主线，从 FA-1 开始就确立了方向。

FA-1 的突破在于发现了问题本质。Attention 在常见形状下往往是内存受限，关键是减少 HBM 访问与中间结果的物化：

```text
# 传统方法（示意）
HBM_access ≈ 4·N·d + 4·N^2  # Q/K/V/O + S/P 读写

# FlashAttention（示意）
# 通过分块在片与在线 softmax，避免 S/P 的 N^2 物化，主项随 N^2 增长但前系数显著更小
# 具体常数取决于 tile 与实现；实验中常见 2–9× 的 IO 降低与端到端加速（中长序列更明显）
```

FA-2 进一步优化了访问模式。通过改变循环顺序和 warp 划分，减少了 SMEM 的读写次数：

```python
# FA-1: Split-K 需要跨 warp 通信
SMEM_transactions_per_iter = 4  # 写部分结果 + 读全局结果

# FA-2: Split-Q 无需跨 warp 通信  
SMEM_transactions_per_iter = 2  # 只读 K/V（广播）

# 对于 16 次 K/V 迭代（示例，精确取决于实现）
FA-1 SMEM 流量: 16 * 4 = 64 transactions
FA-2 SMEM 流量: 16 * 2 = 32 transactions
减少: 2x
```

FA-3 引入硬件异步搬运，彻底解耦了数据移动和计算：

```python
# FA-2: 数据搬运阻塞计算
for kv_block in kv_blocks:
    load_to_smem(kv_block)  # 所有 warp 等待
    __syncthreads()
    compute(kv_block)       # 然后计算

# FA-3: 异步流水线
Producer: load_to_smem(kv_block[i+1])  # 加载下一块
Consumer: compute(kv_block[i])          # 同时计算当前块

# 时间重叠（示例数值，仅用于说明重叠收益）
FA-2: load_time + compute_time = 12μs + 48μs = 60μs
FA-3: max(load_time, compute_time) = max(12μs, 48μs) = 48μs
减少: 1.25x（形状/调度不同会有差异）
```

FA-4 通过 TMEM 进一步减少中间结果在 SMEM 的存储需求，让更多数据可以留在片上。

**主线 2：计算密度的不断提高**

计算密度指的是有效计算占总时间的比例。Attention 中大量的非 matmul 操作（softmax、缩放、累加）会拖累整体性能。

FA-1 基本没有优化这个问题，所有操作都是朴素实现。

FA-2 的关键发现是延迟归一化。我们来看具体的指令数对比：

```python
# FA-1: 每次迭代都归一化
for j in range(T_c):  # 假设 T_c=16
    # ... 计算 P_ij, l_ij ...
    
    # 归一化（标量操作，慢）
    O_i = (alpha * l_i / l_new) * O_i       # B_r * d 次除法和乘法
    O_i += (beta * l_ij / l_new) * P_V      # B_r * d 次除法和乘法
    
    # 每次迭代: 4 * B_r * d 次标量运算
    # 总计: 16 * 4 * 128 * 128 = 1,048,576 次

# FA-2: 只在最后归一化
for j in range(T_c):
    # 只累积，不做除法
    O_i = alpha * O_i + beta * P_V  # 2 * B_r * d 次
    l_i = alpha * l_i + beta * l_ij
    
# 循环结束后
O_i = O_i / l_i  # 1 * B_r * d 次除法

# 总计: 16 * 2 * 128 * 128 + 128 * 128 = 540,672 次
# 减少: 1,048,576 / 540,672 = 1.94x
```

在 A100 上，标量运算的吞吐只有 Tensor Core 的 1/16，所以减少标量运算的收益非常明显。

FA-3 对 softmax 做了进一步优化，通过分离 GEMM 和 softmax 到不同的 warpgroup，让它们可以并行执行。

FA-4 引入快速 exp2 近似，用少量标量/向量 FMA 的多项式近似计算 exp2，并与 UMMA/WGMMA 的矩阵乘在时间上交叠，以隐藏 SFU 低吞吐：

```python
# 传统 exp（调用 SFU）
P = exp(S - m)  # 每个元素一次 SFU 调用

# FA-4 快速 exp2（需行级减最大值与底数转换 e^x=2^(x/ln2)）
P = fast_exp2((S - m) * inv_ln2)  # 以少量 FMA 近似

# 常见收益：在 softmax 子段的内核微基准中可观察到 2–20× 吞吐提升（随形状/流水/数据分布而变）；端到端总体提速以实测为准
```

**主线 3：并行度的逐步扩展**

GPU 性能的关键是喂饱所有的计算单元。每一代 FA 都在扩展并行维度。

FA-1 的并行维度：

```text
并行粒度 = batch_size * num_heads * num_q_blocks

# 例如: batch=1, heads=8, seqlen=2048, B_r=128
num_q_blocks = 2048 / 128 = 16
总并行度 = 1 * 8 * 16 = 128 CTAs

# A100 有 108 SM，每个 SM 可驻留多个 CTA
# 该 grid 规模足以填满 108 个 SM；实际占用率取决于寄存器/SMEM 等资源约束
```

但在短序列或小 batch 时：

```text
# batch=1, heads=8, seqlen=512, B_r=128
num_q_blocks = 512 / 128 = 4
总并行度 = 1 * 8 * 4 = 32 CTAs
占用率 = 32 / 108 ≈ 30%  # 太低了
```

FA-2 改善了这个问题，通过改变反向的循环顺序，让 Q 块之间完全独立：

```text
# FA-1 反向: K/V 块并行
并行度 = batch * heads * num_kv_blocks

# FA-2 反向: Q 块并行
并行度 = batch * heads * num_q_blocks

# 对于相同的序列长度，并行度一样
# 但 FA-2 的调度更灵活，GPU 可以更好地隐藏延迟
```

FA-3 通过 warp 专门化，在单个 CTA 内部也实现了流水线并行：

```text
# FA-2: 所有 warp 做同样的事
所有 warp: 等待数据 → 计算 → 等待数据 → 计算 ...

# FA-3: 不同 warp 做不同的事
Producer warp: 加载[0] → 加载[1] → 加载[2] → ...
Consumer warps:  等待[0] → 计算[0] → 等待[1] → 计算[1] → ...

# 时间利用率更高
```

FA-4 进一步细化了 warp 角色（常见为 5 类，具体随形状/调度调整），并且每个 CTA 处理 2 个 Q tile，提高了 K/V 的复用率。

### 未来的可能方向

基于当前的趋势，我们可以预测几个可能的优化方向。

**方向 1：更激进的量化**

FA-3 已经支持 FP8，下一步可能是 INT8 甚至 INT4。挑战在于 softmax 的数值稳定性：

```text
# FP8 的动态范围
E4M3: 2^-6 到 2^8，约 6 个数量级

# Attention 分数的动态范围
S = Q @ K.T / sqrt(d)
# Q, K 是归一化的，S 的范围约 [-10, 10]
exp(S) 的范围: [exp(-10), exp(10)] = [4.5e-5, 22026]
# 跨越 9 个数量级！FP8 装不下

# 可能的解决方案
1. 分段量化: 对不同值域用不同 scale
2. 对数域计算: 用 log-softmax，减少动态范围
3. 混合精度: matmul 用 INT4，softmax 用 FP8
```

**方向 2：稀疏 Attention 的加速**

真实应用中，Attention 矩阵往往是稀疏的（大部分值接近 0）。如果能提前知道哪些位置重要，可以跳过无用的计算。

FA-1 已经有 Block-Sparse 扩展，但还不够高效。问题在于：

```text
# 静态稀疏（编译时已知模式）
# 例如: 只计算局部窗口 + 固定的全局 token
这种比较好优化，可以预先规划内存布局

# 动态稀疏（运行时决定）
# 例如: Top-K attention，只保留分数最高的 K 个
这种很难优化，因为:
1. 需要先算完整的 S 才知道 Top-K
2. 不规则的访问模式，硬件不友好
```

可能的方向是两阶段方法：

```python
# 阶段 1: 粗粒度筛选（低精度，快速）
S_approx = Q_int4 @ K_int4.T  # INT4 快速估计
top_k_indices = topk(S_approx, k=256)  # 每行保留 256 个

# 阶段 2: 精细计算（高精度，只算有用的）
for selected_index in top_k_indices:
    S_exact[selected_index] = Q_fp16 @ K_fp16[selected_index].T
P = softmax(S_exact)
O = P @ V
```

**方向 3：跨层优化**

当前 FA 优化的是单层 Attention。但 Transformer 有几十层，层与层之间可以做协同优化：

```python
# 当前: 每层独立
for layer in layers:
    Q, K, V = layer.qkv_proj(X)
    X = flash_attn(Q, K, V)  # kernel 1
    X = layer.output_proj(X)  # kernel 2
    X = layer.ffn(X)          # kernel 3
# 总共 3N 次 kernel launch（N 是层数）

# 可能的优化: 融合多层
# 挑战: 激活内存爆炸
# 解决: 选择性重计算 + 流水线并行

fused_transformer_block(X, layers):
    # 融合 attention + FFN
    # 中间激活只保存少量 checkpoint
    # 反向时重计算
```

这需要在编译器层面支持，可能是 XLA 或 TorchInductor 的方向。