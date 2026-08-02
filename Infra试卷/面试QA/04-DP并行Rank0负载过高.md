# 04. DP 并行 Rank 0 负载过高

## 1. 问题

分布式训练或推理采用 DP，各 rank 执行相同模型并处理不同数据分片，但只有
Rank 0 的 CPU、GPU、显存或网络负载明显高于其他 rank。

纯 DP 数据面应基本对称：

```text
各 rank：
  读取本地数据分片
  -> 前向与反向计算
  -> 梯度同步
  -> 参数更新
```

DDP 的梯度 AllReduce 也不是“其他 rank 把梯度全部发给 Rank 0”。因此，
Rank 0 单点过载通常来自额外的控制面工作、中心化数据处理或不均匀的数据分片。

## 2. 方案

### 2.1 先确定哪种资源过载

| 资源 | 重点排查 |
|---|---|
| CPU | DataLoader、tokenization、日志、指标聚合、进程协调 |
| GPU 计算 | token 数、序列长度、动态分支、额外评估任务 |
| GPU 显存 | 输出聚合、完整模型状态、优化器状态、缓存 |
| 网络 | Gather、Broadcast、对象通信、Checkpoint I/O |
| 存储 I/O | Rank 0 集中保存模型、日志和评估结果 |

每个 rank 分阶段记录：

```text
data_time
forward_time
backward_time
communication_time
optimizer_time
checkpoint_time
tokens_per_batch
peak_memory
```

比较各 rank 的 P50、P99 和最大值，定位负载从哪个阶段开始分化。

### 2.2 排查 Rank 0 专属逻辑

重点搜索：

```text
if rank == 0:
if dist.get_rank() == 0:
is_global_zero
main_process_only
```

常见的 Rank 0 专属任务包括：

- 汇总并打印全部训练指标；
- 将完整预测结果 Gather 到 Rank 0；
- 构造完整 `state_dict` 并保存 Checkpoint；
- 执行验证、采样或生成；
- 负责数据读取、预处理，再向其他 rank 分发；
- 维护全局队列、缓存、调度器或 RPC 服务。

### 2.3 数据负载按计算量均衡

不能只保证每个 rank 的样本数相同，还要保证计算量接近。对变长序列，主要负载是：

```text
每个 rank 的 token 数
序列长度分布
padding 后的有效计算量
```

处理方式：

- 使用分布式 Sampler，确保各 rank 获得独立且数量一致的数据分片；
- 每个 epoch 调用 `sampler.set_epoch()`，保持各 rank 的 shuffle 一致；
- 使用 bucketing 或 token-budget batching，按 token 数而不是样本数分 batch；
- 对最后一个不完整 batch 使用一致的 `drop_last` 或补齐策略；
- 动态路由任务同时统计实际激活的 token、专家和计算分支。

### 2.4 去掉中心化数据通路

错误结构：

```text
Storage
  -> Rank 0 读取和预处理
  -> Rank 0 向其他 rank 分发
```

推荐结构：

```text
Storage
  -> 各 rank 独立读取自己的数据分片
  -> 各 rank 本地预处理
```

若共享存储承受不了全部并发读取，应使用分片数据集、本地缓存、节点级缓存或
DataLoader 服务，而不是让 Rank 0 成为全局数据代理。

### 2.5 分布式聚合指标

标量指标应先在各 rank 本地计算，再通过 collective 聚合：

```text
local_sum, local_count
  -> AllReduce
  -> global_mean = global_sum / global_count
```

不要将每个样本的 loss、logit 或预测对象全部 Gather 到 Rank 0。只有最终需要展示
或持久化的少量结果由 Rank 0 输出。

Python 对象通信会触发序列化和 Host 侧处理，应尽量将数据转换为 Tensor 后使用
collective。

### 2.6 Checkpoint 分片保存

错误结构：

```text
所有模型与优化器分片
  -> Gather 到 Rank 0
  -> Rank 0 保存完整 Checkpoint
```

推荐结构：

```text
Rank 0 -> shard 0
Rank 1 -> shard 1
...
Rank N -> shard N
```

采用分布式 Checkpoint：

- 每个 rank 保存自己的模型和优化器分片；
- Manifest 由一个轻量协调者生成；
- Checkpoint 写入与下一轮计算异步重叠；
- 仅在离线导出或部署前合并完整权重。

### 2.7 避免 Rooted Collective 集中到 Rank 0

根据语义替换：

| 中心化操作 | 分布式替代 |
|---|---|
| Gather 全部结果到 Rank 0 | AllGather、分片落盘或分布式后处理 |
| Rank 0 汇总梯度 | AllReduce 或 ReduceScatter |
| Rank 0 分发大数据 | 各 rank 独立读取，必要时 AllToAll |
| Gather Python 对象 | Tensor collective |

Broadcast 本身允许所有 rank 接收同一份小型控制信息；如果 Rank 0 持续生成和广播
大对象，则应拆分数据生产逻辑，而不是只更换通信算法。

### 2.8 拆分控制面与数据面

Rank 0 可以保留以下轻量职责：

- 初始化进程组；
- 发布少量全局配置；
- 输出最终标量指标；
- 管理作业生命周期。

重任务应分离：

- 日志和可视化交给异步进程；
- Checkpoint 交给分布式写入线程或独立服务；
- 数据预处理交给各 rank 或数据服务；
- 评估任务分布到独立进程组；
- 节点内协调采用 local leader，避免全部请求集中到 global Rank 0。

### 2.9 检查 CPU、NUMA 和 GPU 亲和性

如果 Rank 0 没有额外业务逻辑，但仍然过载，需要检查：

- Rank 0 是否与系统守护进程、日志进程共用 CPU 核；
- DataLoader worker 是否全部绑定到同一个 NUMA；
- GPU 与 NIC 是否跨 NUMA；
- Rank 0 是否承担额外的 rendezvous、RPC 或监控进程；
- 各 rank 的 CPU affinity、内存策略和 NIC 绑定是否一致。

### 2.10 不要直接减少 Rank 0 的 batch

直接减少 Rank 0 的 batch 会改变有效 global batch 和梯度权重，还可能导致其他
rank 在同步点等待。应先消除 Rank 0 的额外职责或按 token 重新均衡数据。

如果必须支持不等量 batch，需要同时处理：

- 按实际样本数或 token 数加权梯度；
- 不同步的迭代次数；
- collective 参与一致性；
- DDP Join 或等价的容错机制。

### 2.11 最终结构

```text
各 Rank 数据面
├── 独立读取数据分片
├── 本地预处理
├── 相近的 token / compute budget
├── 相同的前向、反向与优化器流程
└── 使用 collective 对称同步

分布式辅助任务
├── 指标：局部统计后 AllReduce
├── 评估：各 rank 并行后聚合小结果
├── Checkpoint：各 rank 保存分片
├── 日志：异步写入
└── 数据服务：分片读取或节点级缓存

Rank 0 控制面
├── 初始化
├── 少量配置广播
├── 作业状态协调
└── 最终标量输出
```

## 3. 面试追问

### Q1：DP 为什么还会出现 Rank 0 单点过载？

A：

- DP 只保证模型计算和梯度同步基本对称，不保证应用层逻辑对称。
- 日志、数据分发、结果聚合、评估和 Checkpoint 通常被集中到 Rank 0。
- 也可能是数据按样本数均分，但 token 数和实际计算量不均。

### Q2：DDP AllReduce 是否由 Rank 0 汇总全部梯度？

A：

- 不是。
- AllReduce 是所有 rank 共同参与的 collective，每个 rank 最终得到相同的归约结果。
- Rank 0 单点过载通常不应归因于标准 DDP AllReduce。

### Q3：如何最快定位问题？

A：

- 先区分 CPU、GPU、显存、网络和存储 I/O。
- 再比较各 rank 的阶段耗时、token 数和峰值显存。
- 最后检查所有 `rank == 0` 分支和 rooted collective。

### Q4：为什么样本数均匀仍可能负载不均？

A：

- 变长序列的计算量由 token 数、padding 长度和动态分支决定。
- 相同样本数不代表相同 token 数，也不代表相同 FLOPs。
- 应使用 token-budget batching 或长度 bucketing。

### Q5：Rank 0 保存 Checkpoint 太慢怎么办？

A：

- 不在 Rank 0 聚合完整模型和优化器状态。
- 每个 rank 保存自己的 shard，仅集中生成 Manifest。
- 使用异步写入，将 I/O 与后续计算重叠。

### Q6：可以轮换 Rank 0 吗？

A：

- 轮换 root 只能摊平无法消除的轻量中心化任务。
- 对数据加载、结果聚合和 Checkpoint 等持续大流量任务，应改为真正的分布式实现。
- Rank 0 是逻辑角色，不应成为数据面的固定瓶颈。
