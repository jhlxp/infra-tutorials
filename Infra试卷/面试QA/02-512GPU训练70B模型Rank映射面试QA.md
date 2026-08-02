# 02. 512 GPU 训练 70B 模型 Rank 映射

## 1. 问题

使用 512 张 GPU 训练 70B Dense 模型。集群包含 8 个节点，每个节点 64 张
GPU。设计 TP、PP、DP 并行度和 rank 映射，使高频、大流量通信尽量使用节点内
高速互联。

需要明确：

- TP、PP、DP 的并行度；
- global rank 与物理节点、local rank 的映射；
- TP group、DP group、PP group 的构造；
- TP collective、DP 梯度同步和 PP P2P 的物理通信位置；
- pipeline bubble、显存和通信性能的验证方法。

## 2. 方案

### 2.1 并行度

```text
TP = 8
PP = 8
DP = 8

TP × PP × DP = 8 × 8 × 8 = 512
```

- TP=8：单层参数和计算由 8 张 GPU 切分。
- PP=8：模型切成 8 个流水线阶段，每个节点承载一个 stage。
- DP=8：每个 stage 保留 8 份数据并行副本。

### 2.2 Rank order

```text
order = tp-dp-pp
```

TP 是最内层维度，DP 次之，PP 是最外层维度。一个节点正好容纳：

```text
TP × DP = 8 × 8 = 64 ranks
```

因此，一个节点对应一个完整的 PP stage。

### 2.3 Rank 映射

```text
global_rank = tp + TP × (dp + DP × pp)
            = tp + 8 × dp + 64 × pp

node_id     = global_rank / 64
local_rank  = global_rank % 64
```

化简后：

```text
node_id    = pp
local_rank = 8 × dp + tp
```

节点 `pp` 承载流水线阶段 `pp`；节点内每连续 8 个 rank 构成一个 TP group。

### 2.4 通信组

```text
TP group(pp, dp):
  {64 × pp + 8 × dp + tp | tp = 0..7}

DP group(pp, tp):
  {64 × pp + 8 × dp + tp | dp = 0..7}

PP group(dp, tp):
  {64 × pp + 8 × dp + tp | pp = 0..7}
```

- TP group：同一 stage、同一数据副本的 8 个张量分片。
- DP group：同一 stage、同一张量分片的 8 个参数副本。
- PP group：同一数据副本、同一张量分片在 8 个 stage 间的连接。

### 2.5 节点内布局

```text
Node pp
├── local_rank  0..7   -> DP 0, TP 0..7
├── local_rank  8..15  -> DP 1, TP 0..7
├── local_rank 16..23  -> DP 2, TP 0..7
├── local_rank 24..31  -> DP 3, TP 0..7
├── local_rank 32..39  -> DP 4, TP 0..7
├── local_rank 40..47  -> DP 5, TP 0..7
├── local_rank 48..55  -> DP 6, TP 0..7
└── local_rank 56..63  -> DP 7, TP 0..7
```

### 2.6 通信位置

```text
节点内：
  TP AllReduce / ReduceScatter / AllGather
  DP gradient synchronization

跨节点：
  PP activation P2P
  PP activation-gradient P2P
```

TP 通信频率高，DP 梯度同步数据量大，二者放在节点内。跨节点主要承担相邻
PP stage 间结构固定、可与计算流水重叠的 P2P 通信。

### 2.7 Batch 与流水线

```text
global_batch_size
  = micro_batch_size
  × DP
  × gradient_accumulation_steps
```

micro-batch 数量需要覆盖 PP 流水线深度。若 pipeline bubble 仍然明显，可使用
VPP 将每个物理 stage 进一步切成多个 virtual chunks。

### 2.8 验证

- 检查每个进程的 `global_rank`、`node_id`、`local_rank`、`tp_rank`、
  `dp_rank` 和 `pp_rank`。
- 检查 TP group 和 DP group 是否位于节点内，PP group 是否按固定
  `(dp, tp)` 跨节点连接。
- 使用 NCCL topology/debug 信息检查 GPU、NIC 和 collective 路径。
- 使用 Nsight Systems 检查 TP、DP、PP 通信耗时及其与计算的重叠。
- 根据显存、MFU、pipeline bubble 和通信暴露时间调整并行度。

## 3. 面试追问

### Q1：为什么不使用 TP=64？

A：

- TP 越大，每层 collective 的参与范围越大。
- 单卡 GEMM 规模随 TP 增大而缩小，可能降低 Tensor Core 利用率。
- TP 满足模型切分和显存约束即可，不应仅为覆盖节点内全部 GPU 而增大。

### Q2：为什么选择 PP=8？

A：

- PP 将模型参数、梯度和激活分布到 8 个节点。
- 一个节点对应一个 stage，rank 映射稳定。
- 代价是 pipeline bubble，需要使用 micro-batch 和 VPP 控制。

### Q3：为什么选择 `tp-dp-pp`？

A：

```text
tp-dp-pp：
  TP、DP 节点内
  PP 跨节点

tp-pp-dp：
  TP、PP 更容易节点内
  DP 跨节点
```

主方案将高频 TP collective 和大流量 DP 梯度同步留在节点内，仅让 PP P2P
跨节点。

### Q4：VPP 的作用是什么？

A：

- 将每个物理 PP stage 切成多个 virtual chunks。
- 通过更细粒度的交错调度降低 pipeline bubble。
- 代价是增加 stage 切换、P2P 次数和调度复杂度。

### Q5：如果 70B 是 MoE，方案如何变化？

A：

- 增加 EP，世界大小变为 `TP × PP × DP × EP`，必要时再加入 CP。
- EP dispatch/combine 是 AllToAllV，需要考虑 NVLink 域和跨节点 Rail。
- 并行度取决于专家数、TopK、专家容量，以及 70B 指总参数还是激活参数。

### Q6：如何判断方案是否需要调整？

A：

- 显存不足：调整 TP/PP，使用 activation checkpointing 或 distributed optimizer。
- TP 通信暴露：降低 TP 或调整 TP group 的物理位置。
- PP bubble 过大：增加 micro-batch 或使用 VPP。
- PP P2P 过慢：评估 `tp-pp-dp`，同时测量由此引入的跨节点 DP AllReduce。

## 参考资料

- [Megatron Core Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/user-guide/parallelism-guide.html)
- [Megatron-LM parallel_state.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
- [NVIDIA NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
