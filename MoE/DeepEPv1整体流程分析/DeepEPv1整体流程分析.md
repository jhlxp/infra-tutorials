# DeepEP V1整体流程分析

DeepEP V1是面向MoE Expert Parallel的GPU通信库。它的核心不是做专家计算，而是高性能完成MoE里的两类通信：

- `dispatch`：把原始token按`topk_idx`发送到专家所在rank。
- `combine`：把专家输出按原始token顺序归并回来。

所以它在MoE层中的位置是：

```text
gating / router
    │
    v
topk_idx / topk_weights
    │
    v
DeepEP dispatch
    │
    v
expert FFN / GEMM
    │
    v
DeepEP combine
    │
    v
原token顺序的MoE输出
```

本文按“概念 -> 执行流程 -> 模块拆解 -> 设计小结”的方式展开。

## 1. 基本概念

### 1.1 DeepEP V1解决什么问题

MoE中，每个token会被router选到若干个expert。expert通常被切分到不同GPU上，所以token hidden必须发生跨GPU通信。

DeepEP V1解决的是这个通信问题：

```text
输入:
  x             [num_tokens, hidden]
  topk_idx      [num_tokens, num_topk]
  topk_weights  [num_tokens, num_topk]

dispatch后:
  每个rank拿到自己local expert需要处理的token

combine后:
  每个rank拿回属于自己原始token的加权输出
```

DeepEP V1本身不包含专家GEMM。专家计算在`dispatch`和`combine`之间，由框架或模型代码调用其他算子完成。

### 1.2 Dispatch是什么

`dispatch`就是把token送到expert所在rank。

逻辑上看，每个token有`num_topk`个目标expert：

```text
token i
│
├── expert e0 -> rank r0
├── expert e1 -> rank r1
├── expert e2 -> rank r2
└── ...
```

但是DeepEP V1不会简单按TopK record无脑发送hidden。它会先做token-rank去重。

如果一个token命中同一个rank上的多个expert，那么这个token hidden只发一次：

```text
token i:
  expert 10 -> rank 2
  expert 11 -> rank 2
  expert 37 -> rank 5

dispatch发送:
  rank 2: 一份hidden
  rank 5: 一份hidden
```

目标rank收到一份hidden后，通过`recv_topk_idx / recv_topk_weights`知道这个token在本rank上命中了哪些local expert。

### 1.3 Combine是什么

`combine`是`dispatch`的反向过程。

dispatch后，每个rank上的local expert完成计算，产生专家输出。combine要把这些专家输出送回原始token所在rank，并按TopK权重累加。

本质公式是：

```text
output[token] =
    sum_k topk_weights[token, k] * expert_output[token, k]
```

所以combine不只是“把数据发回去”，还要做reduce。

### 1.4 Normal模式和Low-latency模式

DeepEP V1有两套主要通信路径：

| 模式 | 场景 | 设计重点 |
|---|---|---|
| Normal | 训练 / prefill，高吞吐。 | 先统计布局，再紧凑通信；支持NVLink和RDMA两级转发。 |
| Low-latency | decode，小batch，低时延。 | 预留最大buffer，避免动态接收大小等待；按expert-major布局组织数据。 |

normal模式适合token多、数据量大的场景。

low-latency模式适合token少、时延敏感的场景。

### 1.5 Rank拓扑

DeepEP V1把全局rank拆成两个维度：

```text
global_rank = rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank

rdma_rank = global_rank / NUM_MAX_NVL_PEERS
nvl_rank  = global_rank % NUM_MAX_NVL_PEERS
```

源码里：

```text
NUM_MAX_NVL_PEERS = 8
```

含义是：

- `nvl_rank`：服务器内第几张GPU。
- `rdma_rank`：第几个RDMA domain，可以理解为第几个服务器。
- 同一个服务器内的rank通过NVLink / CUDA IPC互访。
- 不同服务器之间，同`nvl_rank`的GPU通过RDMA通信。

以4服务器、每服务器8张GPU为例：

```text
Server 0: rank  0  1  2  3  4  5  6  7
Server 1: rank  8  9 10 11 12 13 14 15
Server 2: rank 16 17 18 19 20 21 22 23
Server 3: rank 24 25 26 27 28 29 30 31

Rail 0: rank 0  -> rank 8  -> rank 16 -> rank 24
Rail 1: rank 1  -> rank 9  -> rank 17 -> rank 25
...
Rail 7: rank 7  -> rank 15 -> rank 23 -> rank 31
```

normal internode通信的关键就是利用这个两级拓扑：

```text
服务器内:
  NVLink / CUDA IPC

服务器间:
  RDMA / NVSHMEM IBGDA
```

### 1.6 Buffer、Channel、Ring Buffer

DeepEP V1的通信不是每次临时申请小buffer，而是在初始化时申请大块通信buffer，然后kernel内部手工切分。

几个核心概念：

| 概念 | 含义 |
|---|---|
| `Buffer runtime` | 当前rank的通信资源对象，保存IPC指针、NVSHMEM buffer、workspace、host counter等。 |
| `NVLink buffer` | 同服务器内GPU通过CUDA IPC互相访问的buffer。 |
| `RDMA buffer` | NVSHMEM symmetric memory，用于跨服务器RDMA。 |
| `Channel` | 软件层面的通信分片。normal模式中`num_channels = num_sms / 2`。 |
| `Ring Buffer` | 每个channel内部通过`head / tail`推进生产消费。 |

normal模式里默认：

```text
Buffer.num_sms = 20
num_channels   = 10
```

每个channel对应一组buffer和一组通信任务。

## 2. 用法

上层使用DeepEP V1时，通常分三步：

1. 创建通信buffer。
2. dispatch前计算layout。
3. 调用dispatch / combine。

典型normal模式代码结构：

```python
buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)

num_tokens_per_rank, \
num_tokens_per_rdma_rank, \
num_tokens_per_expert, \
is_token_in_rank, event = buffer.get_dispatch_layout(
    topk_idx,
    num_experts,
)

recv_x, recv_topk_idx, recv_topk_weights, \
num_recv_tokens_per_expert, handle, event = buffer.dispatch(
    x,
    topk_idx=topk_idx,
    topk_weights=topk_weights,
    num_tokens_per_rank=num_tokens_per_rank,
    num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
    num_tokens_per_expert=num_tokens_per_expert,
    is_token_in_rank=is_token_in_rank,
)

expert_output = run_local_experts(
    recv_x,
    recv_topk_idx,
    recv_topk_weights,
)

combined_x, combined_topk_weights, event = buffer.combine(
    expert_output,
    handle,
    topk_weights=recv_topk_weights,
)
```

这里`handle`很重要。

它保存dispatch时生成的通信路径信息，combine会用它反向归并。

```text
handle不是通信结果缓存，
而是通信路径缓存。
```

## 3. 执行流程

对DeepEP V1来说，流程本质上就是一组通信kernel按阶段完成布局统计、数据搬运、机内转发和结果归并。

所以本节直接把“流程动作”和“对应kernel”放在一起看。

### 3.1 Normal模式整体执行流程

normal模式的核心流程如下：

```text
┌──────────────────────────────────────────────┐
│ 1. Buffer初始化                               │
│    CUDA IPC / NVSHMEM / workspace / counters │
└───────────────────────┬──────────────────────┘
                        │
                        v
┌──────────────────────────────────────────────┐
│ 2. get_dispatch_layout                        │
│    kernel: layout::get_dispatch_layout        │
│    topk_idx -> rank / RDMA rank / expert统计  │
└───────────────────────┬──────────────────────┘
                        │
                        v
┌──────────────────────────────────────────────┐
│ 3. notify_dispatch                            │
│    kernel: intranode::notify_dispatch 或      │
│            internode::notify_dispatch         │
│    交换接收计数，生成prefix matrix            │
│    CPU得到num_recv_tokens并分配recv_x         │
└───────────────────────┬──────────────────────┘
                        │
                        v
┌──────────────────────────────────────────────┐
│ 4. dispatch数据面                             │
│    kernel: intranode::dispatch 或             │
│            internode::dispatch                │
│    NVLink / RDMA / NVLink两级转发              │
│    输出recv_x / recv_topk_idx / recv_weights  │
└───────────────────────┬──────────────────────┘
                        │
                        v
┌──────────────────────────────────────────────┐
│ 5. local expert计算                           │
│    DeepEP V1不负责这一步                      │
└───────────────────────┬──────────────────────┘
                        │
                        v
┌──────────────────────────────────────────────┐
│ 6. cached_notify_combine / cached_notify      │
│    kernel: intranode::cached_notify_combine 或│
│            internode::cached_notify           │
│    复用dispatch handle，准备combine ring      │
└───────────────────────┬──────────────────────┘
                        │
                        v
┌──────────────────────────────────────────────┐
│ 7. combine数据面                              │
│    kernel: intranode::combine 或              │
│            internode::combine                 │
│    反向发送专家输出，按TopK weight reduce     │
└──────────────────────────────────────────────┘
```

这个图是理解DeepEP V1的主线。

真正复杂的是第4步和第7步：它们内部有intranode和internode两种实现。

### 3.2 Intranode dispatch数据面图

intranode表示所有rank都在一个NVLink domain内。

这一步对应`intranode::dispatch`。它把“源rank扫描token”和“目标rank整理recv_x”放在同一个kernel里完成。

```text
Source rank
    │
    ├── sender block
    │       扫描token
    │       判断is_token_in_rank[token, dst_rank]
    │       写目标rank的NVLink channel ring buffer
    │
    v
Destination rank NVLink ring buffer
    │
    ├── receiver block
    │       轮询tail
    │       读取hidden / topk / weight / src_idx
    │       写本地recv_x
    │
    v
recv_x
```

intranode路径的核心是：

```text
发送方直接写目标GPU的IPC buffer；
目标GPU再把ring buffer里的数据整理成连续recv_x。
```

### 3.3 Internode dispatch数据面图

internode表示跨多个RDMA domain。

进入`internode::dispatch`之前，`get_dispatch_layout`和`notify_dispatch`已经准备好了接收计数、prefix和channel offset。真正的数据面从`internode::dispatch`开始：

```text
Source rank上的token
    │
    ├── 1. 按目标服务器去重
    │      一个token命中同一目标服务器内多个expert时，跨机只发送一份hidden
    │      对应: RDMASender
    │
    v
当前Rail的RDMA send buffer
    │
    ├── 2. 写入hidden、scale、topk信息和SourceMeta
    │      SourceMeta记录这个token在目标服务器内需要转发给哪些nvl_rank
    │      对应: RDMASender
    │
    v
跨机RDMA传输
    │
    ├── 3. GPU通过IBGDA发起put
    │      NIC把数据搬到目标服务器同Rail GPU的RDMA recv buffer
    │      对应: RDMASenderCoordinator
    │
    v
目标服务器同Rail GPU
    │
    ├── 4. 解析SourceMeta
    │      判断哪些目标GPU需要这个token
    │      对应: RDMAAndNVLForwarder
    │
    v
目标服务器内部NVLink ring buffer
    │
    ├── 5. 转发到真正目标GPU
    │      对应: RDMAAndNVLForwarder
    │
    v
Destination rank
    │
    └── 6. 整理成连续recv_x / recv_topk_idx / recv_topk_weights
           对应: NVLReceivers
```

关键点：

- 跨服务器RDMA阶段按`rdma_rank`去重。
- 一个token如果命中同一目标服务器内多个GPU，RDMA只传一次。
- 目标服务器内通过`SourceMeta`决定要转发给哪些`nvl_rank`。
- 最终目标rank把数据整理成自己的`recv_x`。

### 3.4 Combine数据面图

combine是dispatch的反向路径，但会做reduce。

这一步对应`intranode::combine`或`internode::combine`。执行前会先跑一次cached notify，用于清理状态、同步ring并复用dispatch阶段的handle。

```text
Local expert output
    │
    v
发送回原source rank
    │
    ├── intranode:
    │       NVLink ring buffer
    │
    └── internode:
            NVLink汇聚 -> RDMA回传 -> 源侧接收
    │
    v
Source rank
    │
    ├── 等待同一token的多路TopK专家输出
    ├── 按topk_weights累加
    └── 写combined_x[token]
```

combine的本质不是拷贝，而是：

```text
communication + reduction
```

### 3.5 Low-latency模式全局流程图

low-latency模式不走normal的“先notify再紧凑分配recv_x”路线。

它预先分配最大容量buffer：

```text
packed_recv_x:
  [num_local_experts,
   num_ranks * num_max_dispatch_tokens_per_rank,
   hidden]
```

整体流程：

```text
┌──────────────────────────────────────────────┐
│ 1. 预分配low-latency RDMA buffer             │
│    expert-major最大容量布局                  │
└───────────────────────┬──────────────────────┘
                        │
                        v
┌──────────────────────────────────────────────┐
│ 2. low_latency_dispatch send phase            │
│    kernel: internode_ll::dispatch             │
│    扫描topk_idx，按expert抢slot并写RDMA buffer │
└───────────────────────┬──────────────────────┘
                        │
                        v
┌──────────────────────────────────────────────┐
│ 3. low_latency_dispatch recv phase            │
│    kernel: internode_ll::dispatch             │
│    读取计数，整理packed_recv_x和layout_range  │
└───────────────────────┬──────────────────────┘
                        │
                        v
┌──────────────────────────────────────────────┐
│ 4. expert计算                                 │
└───────────────────────┬──────────────────────┘
                        │
                        v
┌──────────────────────────────────────────────┐
│ 5. low_latency_combine send / recv phase      │
│    kernel: internode_ll::combine              │
│    回传并按TopK reduce                        │
└──────────────────────────────────────────────┘
```

low-latency牺牲更多buffer空间，换掉动态接收大小等待。

## 4. 流程内部模块分析

### 4.1 Buffer初始化

对应源码：

- `deep_ep/buffer.py::Buffer.__init__`
- `csrc/deep_ep.cpp::Buffer::Buffer`
- `csrc/deep_ep.cpp::Buffer::sync`

`Buffer`初始化时主要做四件事：

```text
1. 计算rank拓扑
   rank -> rdma_rank / nvl_rank

2. 准备NVLink IPC资源
   cudaMalloc本地buffer
   cudaIpcGetMemHandle导出handle
   cudaIpcOpenMemHandle打开peer buffer

3. 准备RDMA资源
   初始化NVSHMEM
   申请symmetric rdma_buffer_ptr
   设置IBGDA相关环境变量

4. 准备辅助资源
   32 MiB workspace
   host mapped counters
   communication stream
```

runtime结构可以抽象为：

```text
Buffer runtime
│
├── topology
│   ├── rank
│   ├── rdma_rank
│   └── nvl_rank
│
├── NVLink IPC pointer table
│   ├── buffer_ptrs
│   └── barrier_signal_ptrs
│
├── RDMA symmetric buffer
│   └── rdma_buffer_ptr
│
├── workspace
│
└── host mapped counters
```

#### 4.1.1 Buffer layout整体视图

DeepEP V1的buffer分两类：

```text
1. 持久通信buffer
   初始化时申请，dispatch/combine反复复用。

2. 本次调用输出tensor
   每次dispatch/combine根据接收规模动态创建，比如recv_x、recv_topk_idx、combined_x。
```

本节主要看第一类，也就是持久通信buffer。

```text
Buffer runtime
│
├── NVLink IPC buffer
│   ├── buffer_ptrs[0..7]
│   │   每个指针指向同一服务器内某个nvl_rank暴露出来的CUDA IPC buffer
│   └── 用于服务器内GPU互写、ring buffer、barrier
│
├── RDMA symmetric buffer
│   ├── rdma_buffer_ptr
│   │   NVSHMEM symmetric memory
│   └── 用于跨服务器RDMA send/recv
│
├── workspace
│   └── 32 MiB临时工作区
│
└── host mapped counters
    ├── moe_recv_counter
    ├── moe_recv_expert_counter
    └── moe_recv_rdma_counter
```

本地NVLink IPC buffer的物理布局是：

```text
buffer_ptrs[nvl_rank]
│
├── communication region
│   大小: num_nvl_bytes
│   用途: notify / dispatch / combine内部按需切分
│
├── barrier_signal[NUM_MAX_NVL_PEERS]
│   类型: int
│   用途: 同服务器内GPU barrier
│
├── buffer_ptrs_gpu[NUM_MAX_NVL_PEERS]
│   类型: void*
│   用途: GPU侧访问同服务器内其他rank的IPC buffer
│
└── barrier_signal_ptrs_gpu[NUM_MAX_NVL_PEERS]
    类型: int*
    用途: GPU侧访问同服务器内其他rank的barrier signal
```

Normal intranode dispatch时，`communication region`会被切成接收侧ring buffer：

```text
buffer_ptrs[receiver_nvl_rank].communication_region
│
├── rank_prefix_matrix
│   shape: [num_ranks, num_ranks]
│   用途: 记录每个src rank到每个dst rank的接收prefix
│
├── channel metadata
│   shape: [num_channels, num_ranks]
│   ├── channel_start_offset
│   ├── channel_end_offset
│   ├── channel_head_idx
│   └── channel_tail_idx
│
├── channel_x_buffers
│   shape: [num_channels, num_ranks, nvl_recv_slots, hidden]
│   用途: token hidden ring buffer
│
├── channel_src_idx_buffers
│   shape: [num_channels, num_ranks, nvl_recv_slots]
│   用途: 原始token index
│
├── channel_topk_idx_buffers
│   shape: [num_channels, num_ranks, nvl_recv_slots, num_topk]
│   用途: 目标rank上的local expert信息
│
├── channel_topk_weights_buffers
│   shape: [num_channels, num_ranks, nvl_recv_slots, num_topk]
│   用途: TopK权重
│
└── channel_x_scales_buffers
    shape: [num_channels, num_ranks, nvl_recv_slots, num_scales]
    用途: FP8 scale
```

这里的关键是：intranode dispatch的ring buffer放在接收方GPU上。

```text
source rank sender block
    │
    ├── 写 buffer_ptrs[dst_rank] 里的 channel_*_buffers
    ├── 推进 channel_tail_idx
    │
    v
destination rank receiver block
    │
    ├── 轮询 channel_tail_idx
    ├── 消费 [head, tail) 范围内的slot
    ├── 推进 channel_head_idx
    │
    v
recv_x / recv_topk_idx / recv_topk_weights
```

Normal internode dispatch有两段buffer：先用RDMA symmetric buffer跨服务器，再用NVLink IPC buffer在目标服务器内转发。

```text
rdma_buffer_ptr
│
├── rdma_channel_data.send
│   shape: [num_channels, num_rdma_ranks, rdma_recv_slots, token_msg]
│
├── rdma_channel_data.recv
│   shape: [num_channels, num_rdma_ranks, rdma_recv_slots, token_msg]
│
├── rdma_channel_meta.send
│   shape: [num_channels, num_rdma_ranks, meta_words]
│
├── rdma_channel_meta.recv
│   shape: [num_channels, num_rdma_ranks, meta_words]
│
├── rdma_channel_head
│   shape: [num_channels, num_rdma_ranks]
│
└── rdma_channel_tail
    shape: [num_channels, num_rdma_ranks]
```

其中一个`token_msg`包含：

```text
token_msg
│
├── hidden
├── x_scales
├── SourceMeta
│   ├── src_rdma_rank
│   └── is_token_in_nvl_rank_bits
├── topk_idx
└── topk_weights
```

目标服务器同Rail GPU收到RDMA数据后，再写目标服务器内部的NVLink ring：

```text
buffer_ptrs[target_nvl_rank].communication_region
│
├── nvl_channel_x
│   shape: [num_channels, num_nvl_ranks, nvl_recv_slots, token_msg]
│
├── nvl_channel_prefix_start
│   shape: [num_channels, num_nvl_ranks, num_rdma_ranks]
│
├── nvl_channel_prefix_end
│   shape: [num_channels, num_nvl_ranks, num_rdma_ranks]
│
├── nvl_channel_head
│   shape: [num_channels, num_nvl_ranks]
│
└── nvl_channel_tail
    shape: [num_channels, num_nvl_ranks]
```

完整的数据面是：

```text
source GPU
    │
    ├── pack token_msg
    │
    v
rdma_channel_data.send
    │
    ├── IBGDA put
    │
    v
remote same-Rail GPU: rdma_channel_data.recv
    │
    ├── 解析SourceMeta
    │
    v
remote server NVLink ring: nvl_channel_x
    │
    v
final destination GPU
    │
    └── 写recv_x
```

Low-latency模式不使用Normal的紧凑ring layout，而是在RDMA symmetric buffer里预留最大容量，并使用双buffer切换：

```text
rdma_buffer_ptr
│
├── signaling buffer 0
│   └── dispatch_recv_count / combine_recv_flag
│
├── signaling buffer 1
│   └── dispatch_recv_count / combine_recv_flag
│
├── send buffer 0
│   └── dispatch_rdma_send_buffer / combine_rdma_send_buffer
│
├── send buffer 1
│   └── dispatch_rdma_send_buffer / combine_rdma_send_buffer
│
├── recv data buffer 0
│   └── dispatch_rdma_recv_data_buffer / combine_rdma_recv_data_buffer
│
└── recv data buffer 1
    └── dispatch_rdma_recv_data_buffer / combine_rdma_recv_data_buffer
```

每次low-latency dispatch/combine选择一个buffer slot，下一次切到另一个slot：

```text
call 0 -> slot 0
call 1 -> slot 1
call 2 -> slot 0
call 3 -> slot 1
```

low-latency dispatch输出的计算侧tensor是expert-major布局：

```text
packed_recv_x
  shape: [num_local_experts,
          num_ranks * num_max_dispatch_tokens_per_rank,
          hidden]

packed_recv_src_info
  shape: [num_local_experts,
          num_ranks * num_max_dispatch_tokens_per_rank]

packed_recv_layout_range
  shape: [num_local_experts, num_ranks]

packed_recv_count
  shape: [num_local_experts]
```

这几个`packed_recv_*`不是持久RDMA buffer，而是本次dispatch返回给上层expert计算使用的tensor。

所以`Buffer`不是后台通信线程，而是当前rank的通信资源对象。只有调用`dispatch / combine`时，runtime才会启动CUDA kernel。

#### 4.1.2 Channel、SM与Warp视图

normal模式里，channel是通信数据面的并行分片。

先把最容易误解的点说清楚：

```text
每次CUDA kernel launch都会创建一组新的grid / block / warp执行实例。
kernel结束后，这批block / warp就退出。

DeepEP V1不是后台常驻一个永远不退出的通信kernel。
dispatch和combine也不是共用同一批还活着的warp。
```

normal模式的一次高层dispatch通常涉及多步：

```text
get_dispatch_layout kernel
    │
    v
notify_dispatch kernel
    │
    v
dispatch数据面kernel
```

normal模式的一次高层combine通常涉及：

```text
cached_notify_combine / cached_notify kernel
    │
    v
combine数据面kernel
```

本节下面讨论的channel、block和warp，主要指`dispatch / combine`数据面kernel，不包含layout/notify这些准备kernel。

还有一个关键点：

```text
intranode dispatch数据面 和 internode dispatch数据面 是二选一，不是同时launch。

如果 get_num_rdma_ranks() == 1:
  dispatch -> intranode::dispatch

如果 get_num_rdma_ranks() > 1:
  dispatch -> internode::dispatch
```

所以多机时不会先launch 20个block跑`intranode::dispatch`，再launch 20个block跑`internode::dispatch`。多机路径只launch一个`internode::dispatch`数据面kernel，它内部同时包含：

```text
RDMA发送
RDMA接收
目标服务器内部NVLink转发
最终recv_x写入
```

combine也是同理：

```text
如果 get_num_rdma_ranks() == 1:
  combine数据面 -> intranode::combine

如果 get_num_rdma_ranks() > 1:
  combine数据面 -> internode::combine
```

normal数据面源码里使用：

```cpp
num_channels = config.num_sms / 2;
```

默认`num_sms = 20`时：

```text
num_channels = 10

channel 0 -> block 0 / block 1
channel 1 -> block 2 / block 3
channel 2 -> block 4 / block 5
...
channel 9 -> block 18 / block 19
```

这里源码变量名叫`sm_id = blockIdx.x`。它不是硬件SM编号，而是逻辑block编号。

DeepEP V1的通信kernel按persistent worker风格写：每个block在kernel生命周期内持续处理一个通信分片。但源码精确控制的是launch多少个block，不是硬绑定到哪一个物理SM。因此本文后面只用`gridDim.x / blockDim.x / warps per block`描述资源组织。

先看源码能确定的数量。这里不写“占用多少硬件SM”，因为CUDA源码只指定`gridDim.x`和`blockDim.x`，实际block调度到哪个SM由GPU调度器决定。

| 数据面kernel | 选择条件 | `gridDim.x` | `blockDim.x` | warps / block | normal channel数 | channel到block映射 |
|---|---|---:|---:|---:|---:|---|
| `intranode::dispatch` | `num_rdma_ranks == 1` | `config.num_sms` | 768 | 24 | `config.num_sms / 2` | `block 2c`发送，`block 2c+1`接收 |
| `intranode::combine` | `num_rdma_ranks == 1` | `config.num_sms` | 768 | 24 | `config.num_sms / 2` | `block 2c`发送，`block 2c+1`接收 |
| `internode::dispatch` | `num_rdma_ranks > 1` | `num_channels * 2` | `(7 + 1 + 8) * 32 = 512` | 16 | `config.num_sms / 2` | `block 2c`做RDMA接收/NVLink转发，`block 2c+1`做RDMA发送/NVLink接收 |
| `internode::combine` | `num_rdma_ranks > 1` | `num_channels * 2` | `(num_forwarder_warps + 1) * 32` | `num_forwarder_warps + 1` | `config.num_sms / 2` | `block 2c`做NVLink发送/RDMA接收，`block 2c+1`做NVLink接收/RDMA发送 |

normal默认配置里：

```text
Buffer.num_sms = 20
num_channels   = config.num_sms / 2 = 10

单GPU单次数据面kernel:
  gridDim.x = 20 blocks
  channel数 = 10
  每个channel = 2个blocks
```

注意：

```text
20是单GPU / 单rank / 单次数据面kernel的launch block总数。
不是每个channel 20个block。
也不是源码硬绑定“正好使用20个物理SM”。
```

如果是4服务器、每服务器8卡，那么32个rank会各自启动自己的数据面kernel：

```text
全局launch block总数 = 32 ranks * 20 blocks/rank = 640 blocks
```

intranode dispatch和intranode combine最简单：一个channel使用两个block。

源码里这两个数据面kernel都使用：

```cpp
kNumThreads = 768;
```

也就是每个block有：

```text
768 threads = 24 warps
```

```text
channel c
│
├── block 2c
│   └── sender block
│       扫描token / expert输出
│       写目标rank的NVLink ring buffer
│       推进tail
│
└── block 2c + 1
    └── receiver block
        轮询tail
        读取NVLink ring buffer
        写recv_x或combined_x
        推进head
```

这些warp没有再被命名成`RDMASender / Forwarder`这种固定角色，而是在block内部按目标rank切片：

```text
responsible_rank = thread_id / num_threads_per_rank
```

所以intranode的warp视图是：

```text
sender block:
  多组warp分别负责不同dst rank
  每组warp扫描本channel负责的token范围

receiver block:
  多组warp分别负责不同src rank
  每组warp消费对应src rank写来的ring buffer
```

所以intranode路径里，dispatch和combine都复用同一类channel结构：

```text
dispatch:
  source token -> channel ring -> destination recv_x

combine:
  expert output -> channel ring -> source combined_x
```

internode dispatch也是`num_channels * 2`个block，但每个block内部按warp role进一步拆分。

默认参数：

```text
kNumDispatchRDMASenderWarps = 7
NUM_MAX_NVL_PEERS           = 8

每个block = 7 + 1 + 8 = 16 warps = 512 threads
```

源码里的role分配是：

```text
channel c
│
├── block 2c
│   ├── RDMAAndNVLForwarder warps x8
│   │   读取RDMA recv buffer
│   │   解析SourceMeta
│   │   转发到目标服务器内部NVLink ring
│   │
│   └── ForwarderCoordinator分支
│       warp_id >= 8进入该分支
│       其中target_rank == 0的warp负责forwarder侧head回收
│       其余warp直接退出
│
└── block 2c + 1
    ├── RDMASender warps x7
    │   扫描source token
    │   pack token_msg
    │   写RDMA send buffer
    │
    ├── RDMASenderCoordinator warp x1
    │   通过IBGDA发起RDMA put
    │   推进远端tail
    │
    └── NVLReceivers warps x8
        读取本服务器NVLink ring
        写最终recv_x / recv_topk_idx / recv_topk_weights
```

这就是internode dispatch的完整并行视图：

```text
RDMASender warps
    │
    v
RDMA send buffer
    │
    v
RDMASenderCoordinator warp
    │
    v
RDMA / IBGDA
    │
    v
RDMAAndNVLForwarder warps
    │
    v
NVLink ring
    │
    v
NVLReceivers warps
    │
    v
recv_x
```

internode combine还是`num_channels * 2`个block，但role变成反向路径：

```text
WarpRole:
  kNVLSender
  kNVLAndRDMAForwarder
  kRDMAReceiver
  kCoordinator
```

它的语义是：

```text
expert output
    │
    v
NVLSender
    │
    v
NVLAndRDMAForwarder
    │
    v
RDMA / IBGDA
    │
    v
RDMAReceiver
    │
    v
combined_x reduce
```

combine的forwarder warp数量不是固定8个，而是根据RDMA rank数量计算：

```cpp
kNumCombineForwarderWarps = 24;
num_warps_per_forwarder = max(24 / num_rdma_ranks, 1);
num_forwarder_warps = num_rdma_ranks * num_warps_per_forwarder;
```

以4服务器、每服务器8卡为例，`num_rdma_ranks = 4`：

```text
num_warps_per_forwarder = 24 / 4 = 6
num_forwarder_warps     = 4 * 6 = 24

每个block = 24 + 1 = 25 warps = 800 threads
```

对应到一个channel：

```text
channel c
│
├── block 2c
│   ├── NVLSender warps x8
│   │   从本服务器内部NVLink侧收集expert output
│   │
│   ├── RDMAReceiver warps x16
│   │   接收远端RDMA回来的combine数据
│   │
│   └── Coordinator warp x1
│
└── block 2c + 1
    ├── NVLAndRDMAForwarder warps x24
    │   把NVLink汇聚来的数据写入RDMA send buffer并发往源服务器
    │
    └── Coordinator warp x1
```

所以normal internode模式下，可以这样记：

```text
channel = 数据面并行分片
block   = channel里的一个常驻通信worker
warp    = block内部的具体角色
QP      = internode RDMA通道资源；主数据put使用channel_id，部分head回收使用channel_id + num_channels
```

low-latency模式不使用normal这套`num_sms / 2` channel结构。

它按expert数量和`num_device_sms`计算block数，每个block内部再按warp group执行send / recv phase：

```cpp
num_warp_groups     = ceil_div(num_experts, num_device_sms);
num_warps_per_group = 32 / num_warp_groups;
num_warps           = num_warp_groups * num_warps_per_group;
```

```text
low-latency:
  block = 一个低时延通信worker
  warp group = token扫描、slot分配、RDMA写入、接收整理的并行单元
```

所以：

```text
normal:
  num_sms = 20 -> num_channels = 10 -> 每个channel两个block

low-latency:
  不复用normal的num_channels = num_sms / 2结构
  block数由expert数量和token数量计算
```

### 4.2 get_dispatch_layout

对应源码：

- `deep_ep/buffer.py::Buffer.get_dispatch_layout`
- `csrc/kernels/layout.cu::get_dispatch_layout`

输入：

```text
topk_idx: [num_tokens, num_topk]
num_experts
```

输出：

| 输出 | 含义 |
|---|---|
| `num_tokens_per_rank` | 当前rank有多少token需要发往每个目标rank。 |
| `num_tokens_per_rdma_rank` | 当前rank有多少token需要发往每个目标RDMA domain。 |
| `num_tokens_per_expert` | 当前rank发往每个expert的TopK命中次数。 |
| `is_token_in_rank` | 每个token是否命中某个rank。 |

`get_dispatch_layout`分两类统计：

```text
统计expert:
  topk_idx[token, k] -> expert_id
  num_tokens_per_expert[expert_id]++

统计rank / RDMA rank:
  expert_id -> rank_id
  expert_id -> rdma_rank_id
  对每个token做rank去重
```

核心不是“多少TopK record发到rank”，而是“多少token需要发到rank”。

```text
token-rank去重:

token i命中rank 3上的多个expert
    │
    v
is_token_in_rank[i, 3] = true
    │
    v
hidden只发送一份
```

这一步决定了后面dispatch的通信量。

### 4.3 notify_dispatch

normal模式需要先知道接收端要收多少token，才能分配紧凑输出`recv_x`。

所以dispatch真正搬数据前，会先启动`notify_dispatch`。

#### 4.3.1 Intranode notify

对应源码：

- `csrc/kernels/intranode.cu::notify_dispatch`

流程：

```text
每个rank已有:
  num_tokens_per_rank
  num_tokens_per_expert
  is_token_in_rank

notify_dispatch:
  1. NVLink peer之间barrier
  2. 每个rank把自己的发送计数写到buffer
  3. 本rank读取所有peer写来的计数
  4. 得到自己会接收多少token
  5. 得到每个local expert会接收多少token
  6. 生成rank_prefix_matrix和channel_prefix_matrix
```

产物：

```text
rank_prefix_matrix[src_rank, dst_rank]
channel_prefix_matrix[dst_rank, channel_id]
num_recv_tokens_per_expert
```

CPU会读取host mapped counter，拿到`num_recv_tokens`后分配`recv_x`。

#### 4.3.2 Internode notify

对应源码：

- `csrc/kernels/internode.cu::notify_dispatch`

internode notify还要多处理RDMA层：

```text
rdma_channel_prefix_matrix
    每个channel发往每个RDMA rank的prefix

gbl_channel_prefix_matrix
    每个channel发往每个global rank的prefix

recv_rdma_rank_prefix_sum
    当前rank从每个RDMA rank接收的prefix

recv_gbl_rank_prefix_sum
    当前rank从每个global rank接收的prefix
```

可以理解为：

```text
notify_dispatch = 为后续数据面生成路线图和偏移表。
```

### 4.4 Normal intranode dispatch

对应源码：

- `csrc/kernels/intranode.cu::dispatch`

intranode dispatch的kernel grid中：

```text
num_channels = num_sms / 2

block 2c     -> channel c sender
block 2c + 1 -> channel c receiver
```

发送侧：

```text
Sender block
│
├── 根据channel_id切分token范围
├── 遍历token
├── 对每个目标rank检查is_token_in_rank
├── 等待目标rank ring buffer有空slot
├── 写:
│   ├── hidden x
│   ├── src_idx
│   ├── local topk_idx
│   ├── topk_weights
│   └── x_scales
└── release tail
```

接收侧：

```text
Receiver block
│
├── 根据rank_prefix_matrix得到recv_x偏移
├── 轮询channel_tail_idx
├── 从ring buffer读取数据
├── 写recv_x / recv_src_idx / recv_topk_idx / recv_topk_weights
└── 更新channel_head_idx
```

ring buffer控制字段：

```text
channel_start_offset
channel_end_offset
channel_head_idx
channel_tail_idx
```

`recv_x`第一维是当前rank接收到的去重token数量：

```text
recv_x[row]                 一份token hidden
recv_topk_idx[row, :]       该token在本rank命中的local expert
recv_topk_weights[row, :]   对应权重，不属于本rank的位置为0
recv_src_idx[row]           原始source token index
```

### 4.5 Normal intranode combine

对应源码：

- `csrc/kernels/intranode.cu::cached_notify_combine`
- `csrc/kernels/intranode.cu::combine`

combine复用dispatch返回的handle。

```text
dispatch handle
│
├── rank_prefix_matrix
├── channel_prefix_matrix
├── recv_channel_prefix_matrix
├── recv_src_idx
├── is_token_in_rank
└── send_head
```

intranode combine实际使用的是第三个矩阵`recv_channel_prefix_matrix`作为回传方向的`channel_prefix_matrix`：

```python
rank_prefix_matrix, _, channel_prefix_matrix, src_idx, is_recv_token_in_rank, send_head = handle
```

combine流程：

```text
local expert output
    │
    v
sender block
    │
    ├── 根据recv_src_idx找到原始token位置
    ├── 根据send_head找到回传slot
    └── 写回source rank ring buffer
    │
    v
receiver block
    │
    ├── 等待同一token的多个专家输出到齐
    ├── 读取多路结果
    ├── 按topk_weights累加
    └── 写combined_x
```

这里的核心是：

```text
combine = 反向通信 + TopK reduce
```

### 4.6 Normal internode dispatch

对应源码：

- `csrc/kernels/internode.cu::dispatch`

internode dispatch一个channel内部有多个warp role。

源码中定义：

```cpp
enum class WarpRole {
    kRDMASender,
    kRDMASenderCoordinator,
    kRDMAAndNVLForwarder,
    kForwarderCoordinator,
    kNVLReceivers
};
```

全局看，它分成三段：

```text
1. 当前rank pack并发RDMA
2. 目标同Rail rank接收RDMA并做NVLink forward
3. 目标rank从NVLink ring buffer整理出recv_x
```

#### 4.6.1 RDMASender

`RDMASender`扫描本channel负责的token。

对每个token，它判断目标RDMA rank：

```text
token i
│
├── 命中目标RDMA rank 0 -> 写一份
├── 命中目标RDMA rank 1 -> 写一份
└── 没命中的RDMA rank -> 不写
```

写入RDMA send buffer的内容：

```text
hidden x
x_scales
SourceMeta
topk_idx
topk_weights
```

`SourceMeta`保存：

```text
src_rdma_rank
is_token_in_nvl_rank_bits
```

其中`is_token_in_nvl_rank_bits`表示目标服务器内哪些`nvl_rank`需要这个token。

#### 4.6.2 RDMASenderCoordinator

`RDMASender`负责pack，`RDMASenderCoordinator`负责发起RDMA put。

```text
RDMASender
    │
    ├── pack token
    └── 更新共享内存中的pack进度

RDMASenderCoordinator
    │
    ├── 观察RDMASender的pack进度
    ├── 按chunk发起nvshmemi_ibgda_put_nbi_warp
    └── 更新远端rdma_channel_tail
```

它把“数据准备”和“发起RDMA”拆开，但仍然放在同一个CUDA kernel里。

#### 4.6.3 RDMAAndNVLForwarder

目标同Rail rank收到RDMA数据后，读取`SourceMeta`，决定往本服务器内哪些GPU转发。

```text
RDMA recv buffer
    │
    v
读取SourceMeta
    │
    ├── bit 0 -> nvl_rank 0
    ├── bit 1 -> nvl_rank 1
    ├── ...
    └── bit 7 -> nvl_rank 7
    │
    v
写目标NVLink ring buffer
```

这一步就是目标侧的NVLink修复。

#### 4.6.4 NVLReceivers

`NVLReceivers`是最终目标rank上的接收角色。

它从本rank的NVLink ring buffer读取数据，写入紧凑输出：

```text
nvl_channel_x ring buffer
    │
    ├── hidden x
    ├── x_scales
    ├── SourceMeta
    ├── topk_idx
    └── topk_weights
    │
    v
recv_x / recv_x_scales / recv_src_meta / recv_topk_idx / recv_topk_weights
```

同时，它会把global expert id改写成local expert id：

```text
global expert属于当前rank:
    recv_topk_idx = local_expert_id

global expert不属于当前rank:
    recv_topk_idx = -1
    recv_topk_weights = 0
```

### 4.7 Normal internode combine

对应源码：

- `csrc/kernels/internode.cu::cached_notify`
- `csrc/kernels/internode.cu::combine`
- `csrc/kernels/internode.cu::combine_token`

internode combine的warp role：

```cpp
enum class WarpRole {
    kNVLSender,
    kNVLAndRDMAForwarder,
    kRDMAReceiver,
    kCoordinator
};
```

反向流程：

```text
expert output rank
    │
    v
NVLSender
    │
    ├── 写本服务器NVLink ring buffer
    │
    v
NVLAndRDMAForwarder
    │
    ├── 从多个NVLink rank收集
    ├── 对需要跨服务器的数据发RDMA
    │
    v
RDMAReceiver
    │
    ├── 等待远端数据
    ├── 调用combine_token
    └── 写combined_x
```

`combine_token`做的是对同一个原始token的多路结果求和。

```text
读取多个rank发回来的expert output
    │
    v
乘以topk_weights
    │
    v
累加
    │
    v
写combined_x[token]
```

### 4.8 Low-latency dispatch

对应源码：

- `deep_ep/buffer.py::Buffer.low_latency_dispatch`
- `csrc/kernels/internode_ll.cu::dispatch`

low-latency dispatch不使用normal模式的紧凑`recv_x`布局，而是使用expert-major最大buffer：

```text
packed_recv_x:
  [num_local_experts,
   num_ranks * num_max_dispatch_tokens_per_rank,
   hidden]

packed_recv_src_info:
  [num_local_experts,
   num_ranks * num_max_dispatch_tokens_per_rank]

packed_recv_layout_range:
  [num_local_experts,
   num_ranks]
```

kernel内部有send phase和recv phase：

```text
send phase
│
├── 扫描topk_idx
├── 对目标expert atomic抢slot
├── 可选BF16 -> FP8量化
├── 写目标rank的RDMA recv buffer
└── 写expert接收计数

recv phase
│
├── 等待src rank写来的计数
├── 整理packed_recv_x
├── 写packed_recv_src_info
└── 写packed_recv_layout_range
```

low-latency模式的最佳路径要求：

```text
num_qps_per_rank = num_local_experts
```

这样不同local expert可以走独立QP，减少互相阻塞。

### 4.9 Low-latency combine

对应源码：

- `deep_ep/buffer.py::Buffer.low_latency_combine`
- `csrc/kernels/internode_ll.cu::combine`

low-latency combine复用dispatch返回的handle：

```text
packed_recv_src_info
packed_recv_layout_range
num_max_dispatch_tokens_per_rank
hidden
num_experts
```

流程：

```text
send phase
│
├── 根据layout_range遍历每个local expert输出
├── 根据src_info找到原始source token
├── 写回目标rank RDMA recv buffer
└── 写完成flag

recv phase
│
├── 等待TopK结果到齐
├── 读取多个expert输出
├── 按topk_weights累加
└── 写combined_x
```

low-latency API还支持hook式overlap：

```text
先启动send phase
    │
    ├── 上层可以插入计算
    │
    v
hook触发recv phase
```

## 5. Normal与Low-latency对比

| 维度 | Normal模式 | Low-latency模式 |
|---|---|---|
| 目标场景 | 训练 / prefill，高吞吐。 | decode，小batch，低时延。 |
| 输出布局 | 紧凑`recv_x`。 | expert-major最大容量buffer。 |
| dispatch前是否需要layout | 需要`get_dispatch_layout`。 | kernel内部扫描TopK，不走normal layout。 |
| 是否等待接收大小 | 非cached normal路径会等待host mapped counter。 | 预分配最大buffer，避免动态等待。 |
| 通信路径 | intranode NVLink；internode NVLink + RDMA。 | 主要RDMA，可利用P2P路径。 |
| 通信block控制 | `Buffer.set_num_sms`控制normal数据面kernel的launch block数量。 | 没有同样的`Buffer.num_sms`控制路径，block数由LL wrapper按expert/token规模计算。 |
| buffer开销 | 更紧凑。 | 更大。 |
| 适合阶段 | token多、吞吐优先。 | token少、时延优先。 |

一句话：

```text
Normal模式用“先统计，再紧凑通信”换吞吐；
Low-latency模式用“最大buffer，少等待”换时延。
```

## 6. TMA在DeepEP V1中的作用

DeepEP V1在SM90路径里会用TMA，但它不是为WGMMA服务。

DeepGEMM里的TMA通常是：

```text
HBM -> shared memory -> WGMMA
```

DeepEP V1没有GEMM计算，它的TMA更多是通信kernel内部的搬运优化：

```text
global / peer buffer
    │
    v
shared memory staging
    │
    v
global / peer buffer
```

典型位置：

- intranode dispatch receiver从channel buffer搬到`recv_x`。
- internode dispatch forwarder从RDMA buffer搬到NVLink buffer。
- internode dispatch NVL receiver从NVLink buffer搬到`recv_x`。
- combine中的部分reduce路径使用TMA做分段搬运。

所以这里TMA的意义是降低大块数据搬运的warp指令开销，而不是驱动Tensor Core计算。

## 7. 设计小结

### 7.1 token-rank去重

DeepEP V1不是按TopK record复制hidden。

它先把TopK映射到rank：

```text
topk_idx[token, k]
    │
    v
expert_id
    │
    v
rank_id
```

然后生成：

```text
is_token_in_rank[token, rank]
```

同一个token在同一rank命中多个expert，只传一份hidden。

### 7.2 RDMA rank去重

internode dispatch时，还会按RDMA rank去重。

如果一个token命中目标服务器内多个GPU：

```text
RDMA阶段:
  只传一份hidden到目标同Rail rank

目标服务器内部:
  再用NVLink转发到多个destination rank
```

这降低了跨服务器RDMA流量。

### 7.3 两级数据面

normal internode的数据面是：

```text
RDMA负责跨服务器
NVLink负责服务器内部修复
```

dispatch和combine方向不同：

```text
Dispatch:
  RDMA -> 目标侧NVLink分发

Combine:
  源侧NVLink汇聚 -> RDMA回传
```

### 7.4 GPU-initiated通信

internode路径中，GPU kernel直接调用IBGDA device API：

```text
nvshmemi_ibgda_put_nbi_warp
nvshmemi_ibgda_amo_nonfetch_add
nvshmemi_ibgda_quiet
```

这意味着发送、计数更新、tail推进都可以在GPU侧完成，减少CPU proxy参与。

### 7.5 单kernel内多角色流水线

normal internode dispatch不是多个小kernel串起来，而是在一个kernel里安排多类warp：

```text
RDMASender
RDMASenderCoordinator
RDMAAndNVLForwarder
ForwarderCoordinator
NVLReceivers
```

它们通过ring buffer、head/tail、barrier、atomic和polling协作。

代价是占用部分SM和warp资源。

收益是减少CPU参与、减少kernel launch间隙，并把pack、发送、forward、receive融合到一个通信数据面里。
