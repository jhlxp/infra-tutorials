# 03. 512 GPU 训练 1T-A30B MoE 并行设计

## 1. 问题

使用 512 张 GPU 训练 1T-A30B MoE 模型。集群包含 8 个节点，每个节点 64 张
GPU。设计 TP、PP、EP、DP 并行度和 rank 映射，使高频、大流量通信尽量使用
节点内高速互联。

这里 `A30B` 表示每个 token 激活约 30B 参数：

```text
total parameters     ≈ 1T
activated parameters ≈ 30B / token
```

需要明确：

- TP、PP、EP、DP 的并行度；
- global rank 与物理节点、local rank 的映射；
- TP group、EP group、DP group、PP group 的构造；
- MoE dispatch/combine、TP collective、梯度同步和 PP P2P 的物理通信位置；
- expert load balance、pipeline bubble、显存和通信性能的验证方法。

## 2. 方案

### 2.1 并行度

```text
TP = 8
PP = 8
EP = 4
DP = 2

TP × PP × EP × DP = 8 × 8 × 4 × 2 = 512
```

- TP=8：attention、dense MLP 和 expert GEMM 做 8 路张量并行。
- PP=8：模型切成 8 个流水线阶段，每个节点承载一个 stage。
- EP=4：每个 stage 内的 experts 分到 4 个 expert ranks。
- DP=2：expert 参数保留 2 份数据并行副本。

一个节点正好容纳：

```text
TP × EP × DP = 8 × 4 × 2 = 64 ranks
```

因此，一个节点对应一个完整 PP stage，TP、EP 和 expert DP 都放在节点内。

### 2.2 MoE 里的 DP 口径

MoE 中需要区分两种 DP：

```text
dense / attention 参数有效 DP = EP × DP = 4 × 2 = 8
expert 参数有效 DP            = DP = 2
```

原因是：

- EP rank 之间处理不同 token，再通过 MoE dispatch 交换到目标 expert；
- attention、router、norm 等非 expert 参数在 EP 和 DP 维度上都有副本；
- expert 参数只在相同 expert 副本之间同步。

因此 global batch 按 dense 参数的有效 DP 计算：

```text
global_batch_size
  = micro_batch_size
  × EP
  × DP
  × gradient_accumulation_steps
```

### 2.3 Rank order

```text
order = tp-ep-dp-pp
```

TP 是最内层维度，EP 次之，DP 再次，PP 是最外层维度。

```text
global_rank = tp + TP × (ep + EP × (dp + DP × pp))
            = tp + 8 × (ep + 4 × (dp + 2 × pp))

node_id     = global_rank / 64
local_rank  = global_rank % 64
```

化简后：

```text
node_id    = pp
local_rank = tp + 8 × ep + 32 × dp
```

节点 `pp` 承载流水线阶段 `pp`。

### 2.4 节点内布局

```text
Node pp
├── local_rank  0..7    -> DP 0, EP 0, TP 0..7
├── local_rank  8..15   -> DP 0, EP 1, TP 0..7
├── local_rank 16..23   -> DP 0, EP 2, TP 0..7
├── local_rank 24..31   -> DP 0, EP 3, TP 0..7
├── local_rank 32..39   -> DP 1, EP 0, TP 0..7
├── local_rank 40..47   -> DP 1, EP 1, TP 0..7
├── local_rank 48..55   -> DP 1, EP 2, TP 0..7
└── local_rank 56..63   -> DP 1, EP 3, TP 0..7
```

### 2.5 通信组

```text
TP group(pp, dp, ep):
  {64 × pp + 32 × dp + 8 × ep + tp | tp = 0..7}

EP group(pp, dp, tp):
  {64 × pp + 32 × dp + 8 × ep + tp | ep = 0..3}

Dense DP group(pp, tp):
  {64 × pp + 32 × dp + 8 × ep + tp | dp = 0..1, ep = 0..3}

Expert DP group(pp, ep, tp):
  {64 × pp + 32 × dp + 8 × ep + tp | dp = 0..1}

PP group(dp, ep, tp):
  {64 × pp + 32 × dp + 8 × ep + tp | pp = 0..7}
```

- TP group：同一 stage、同一 DP、同一 EP rank 的 8 个张量分片。
- EP group：同一 stage、同一 DP、同一 TP 分片的 4 个 expert ranks。
- Dense DP group：非 expert 参数的 8 个副本。
- Expert DP group：同一 expert 分片的 2 个副本。
- PP group：固定 `(dp, ep, tp)` 跨 8 个 stage 连接。

### 2.6 通信位置

```text
节点内：
  TP AllReduce / ReduceScatter / AllGather
  EP dispatch / combine AllToAllV
  Dense 参数梯度同步
  Expert 参数梯度同步

跨节点：
  PP activation P2P
  PP activation-gradient P2P
```

TP 和 EP 都是高频通信，放在节点内。跨节点只保留 PP P2P，通信结构和上一题
Dense 70B 一样清晰。

### 2.7 Batch 与流水线

```text
global_batch_size
  = micro_batch_size
  × 8
  × gradient_accumulation_steps
```

micro-batch 数量需要覆盖 `PP=8` 的 pipeline bubble。若 bubble 明显，可使用 VPP，
但 VPP 会增加 stage 切换和 P2P 次数，需要实测。

### 2.8 Expert 放置

如果每层有 `E` 个 experts：

```text
num_local_experts_per_ep_rank = E / EP = E / 4
```

expert placement 要保证：

- 每个 EP rank 的 expert 数量接近；
- 热 expert 尽量分散；
- top-k 路由后的 token 数不过度倾斜；
- dropless MoE 下最忙 expert 不拖慢整层。

### 2.9 可选调整

```text
TP=8, PP=8, EP=8, DP=1
```

- expert 参数更分散；
- 适合 expert 显存压力大；
- 代价是 expert DP 变小，global batch 依赖梯度累积。

```text
TP=8, PP=8, EP=2, DP=4
```

- expert AllToAll fanout 更小；
- global batch 更容易做大；
- 代价是每个 EP rank 持有更多 experts。

```text
TP=4, PP=8, EP=4, CP=2, DP=2
```

- 长上下文时加入 CP；
- 代价是 attention 通信更复杂。

### 2.10 验证

- 检查每个进程的 `global_rank`、`node_id`、`local_rank`、`tp_rank`、
  `ep_rank`、`dp_rank` 和 `pp_rank`。
- 检查 TP group、EP group、Dense DP group、Expert DP group 和 PP group 是否符合预期。
- 使用 NCCL topology/debug 信息检查 GPU、NIC 和 collective 路径。
- 使用 Nsight Systems 检查 TP、EP、DP、PP 通信耗时及其与计算的重叠。
- 统计每层每 expert 的 token histogram、drop rate、aux loss 和最忙 rank 时间。
- 根据显存、MFU、pipeline bubble、A2A 暴露时间和 expert load balance 调整并行度。

## 3. 面试追问

### Q1：为什么 MoE 不能直接套 Dense 70B 的 `TP=8, PP=8, DP=8`？

A：

- Dense 没有 expert dispatch/combine。
- MoE 总参数主要在 experts，需要 EP 分散 expert 参数和计算。
- 如果没有 EP，expert 参数会在 DP 副本内重复存放，显存压力很大。
- MoE 的关键不是只放下 active 30B，而是放下 total 1T 并处理 token shuffle。

### Q2：为什么 EP 放在节点内？

A：

- MoE dispatch/combine 每个 MoE 层都会发生。
- AllToAllV 的消息大小随 router 动态变化，容易出现长尾。
- 放在节点内可以利用高速互联，避免每层都跨节点 token shuffle。
- 跨节点只保留 PP P2P，结构更稳定。

### Q3：为什么 dense 参数 DP 是 `EP × DP`，expert 参数 DP 是 `DP`？

A：

- EP rank 处理不同 token，因此非 expert 参数在 EP 维度上也是数据副本。
- Expert 参数被 EP 分片，不同 EP rank 持有不同 experts。
- 同一个 expert 只在 DP 维度上有副本。
- 所以需要区分 Dense DP group 和 Expert DP group。

### Q4：如果 expert 负载不均怎么办？

A：

- 先看每层每 expert token histogram。
- 调整 load balance loss、router z-loss 和 expert capacity。
- 检查 dropless MoE 下最热 expert 是否拖慢整层。
- 必要时做 expert placement，让热 experts 分散到不同 EP rank。

### Q5：什么时候增加 EP？

A：

- Expert 参数放不下。
- 单个 EP rank 的 expert GEMM 太重。
- Hot expert 导致局部 rank 长尾。
- 但增加 EP 会减少 DP 或增加 A2A 复杂度，需要和 global batch 一起看。

### Q6：什么时候增加 DP、降低 EP？

A：

- Expert 参数能放下。
- AllToAllV 成为瓶颈。
- global batch 太小。
- 训练稳定性需要更多数据并行副本。

## 参考资料

- [Megatron Core Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/user-guide/parallelism-guide.html)
- [Megatron-LM parallel_state.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py)
- [Megatron-LM MoE README](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/moe/README.md)
- [DeepSpeed MoE Tutorial](https://www.deepspeed.ai/tutorials/mixture-of-experts/)
- [NVIDIA NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
