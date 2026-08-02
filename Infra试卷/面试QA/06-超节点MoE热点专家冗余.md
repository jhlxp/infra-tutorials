# 06. 超节点 MoE 热点专家冗余

## 1. 简历表述

面向某企业 A3 超节点中多租户资源碎片化场景，针对 DeepSeek-V3 训练 / prefill 中
MoE AllToAll 热点和专家 placement 失衡问题，开发超节点级性能建模与瓶颈分析流程。
基于某企业侧采集的 DSV3 专家访问分布，以及 attention / FFN 在 A3 上的实测计算
时间，将 MoE dispatch、专家计算、combine 和超节点 EPS/OCS 拓扑统一建模；
进一步评估热点专家冗余复制与源端分流方案，用网络传输换取最热服务器上的专家
计算削峰，最终将 MoE 层端到端 CCT 降低 XX%。

- 背景：A3 超节点面向 8 卡服务器和多机高速互联，真实资源池存在多租户切分，作业
  拿到的服务器 / NIC / 网络路径不一定连续，MoE 专家放置容易和物理拓扑
  错配。
- Case：使用某企业运行 DSV3 采集到的专家热度分布作为输入，观察到部分 MoE
  层存在明显热点，最热专家的 token 负载可达到平均值的十倍以上，导致目的端
  服务器同时出现 AllToAll incast 和 FFN 计算长尾。
- 方案：在 A3 超节点拓扑上建模 dispatch / combine 网络时间，并注入某企业采集的
  attention / FFN 计算 profile；对最热专家做少量冗余副本，将部分 token 通过
  网络转发到计算更空闲的服务器执行。
- 结果：在不改变逻辑专家 id 和模型语义的前提下，最热服务器计算负载下降，
  DeepEP overlap 后暴露在关键路径上的专家计算被削弱，端到端 MoE CCT 降低
  XX%。某企业实测部分我主要做技术支持，包括参数配置、指标定义和结果分析。

## 2. Case 参数

### 2.1 硬件目标：Atlas 800T A3

公开规格里，Atlas 800T A3 是面向大模型预训练、后训练的 10U 训练服务器，单机
8 颗昇腾 910；单机片上内存为 `8 × 128GB`，内存带宽 `3.2TB/s`，FP16 算力最高约
`6.0 PFLOPS`，D2D 双向互联带宽 `784GB/s`，网络侧提供 `8 × 400GE` RoCE 直出
接口。

这个 case 的基本对象是一台 A3 服务器，也就是一个 8-NPU tray。超节点只作为
placement 和网络路径的背景：多租户切分后，作业拿到的多台 A3 不一定连续，热点
专家放在哪台 A3 上会直接改变 dispatch/combine 的目的端热点。

### 2.2 模型和 workload

模型按 DSV3 训练 / prefill 的大 token batch 口径建模，不按 decoder 单 token
TPOT 口径建模：

```text
model: DeepSeek-V3 / DSV3 MoE
total params: 671B
activated params: 37B / token
MoE 路由专家: 256
top-k: 8
hidden size: 7168
prompt length: 4096
total tokens/NPU: 16384
dtype: BF16/FP16 or BF8/FP8 profile, 最终以 A3 实测为准
```

这里的核心不是模型参数量本身，而是每个 MoE 层都会发生：

```text
router
  -> dispatch AllToAllV
  -> expert FFN compute
  -> combine AllToAllV
```

训练和 prefill 都是大 token batch：每层的 dispatch/combine 数据量远大于 decode
单 token 场景，FFN 计算也更容易被热点专家的 token 倾斜放大。只要 router 分布
不均，热点专家所在 A3 服务器就会同时承担更多入方向通信和更多 FFN 计算。

### 2.3 某企业采集数据

某企业侧运行 DSV3，采集了每层专家热度分布。我们使用这份真实分布作为
placement 和通信矩阵输入，而不是假设 token 均匀落到所有专家。

采集数据表现出明显热点：

```text
MoE layers: 58
EP experts per layer: 256
observed hotness: 最热专家通常是平均负载的 11x-22x
average max/mean over layers: 约 14.5x
worst layer max/mean: 约 22.7x
```

这说明问题不是 AllToAllV 算法天然慢，而是 DSV3 的真实专家访问分布会把部分
流量集中到少数逻辑专家；如果这些专家的物理位置又集中在同一批 A3 服务器上，
就会形成服务器级热点。

### 2.4 计算 profile

计算 profile 只做一件事：把 A3 上 attention 和 FFN 的实测时间注入模型，判断热点
专家的计算长尾能不能用网络分流削掉。

采集配置：

```text
hardware: A3 server, 8 NPU
workload: DSV3 training / prefill
prompt length: 4096
total tokens/NPU: 16384
hidden: 7168
experts: 256
top-k: 8
EP: 32
rank: 1 NPU
local experts/rank: 8
A3 server: 8 ranks, 64 local experts
avg tokens/expert: 512
hot tokens/expert: 约 5.6K-11.3K
expert-token assignments/rank: 16384 * 8 = 131072
DeepEP dtype:
  dispatch: FP8
  combine: BF16
dispatch payload:
  FP8: 7168B hidden + 224B scale + 16B meta = 7408B/expert-token
  total: 926 MiB/rank, 28.94 GiB/EP32
combine payload:
  BF16: 7168 * 2B = 14336B = 14 KiB/expert-token
  total: 1.75 GiB/rank, 56 GiB/EP32
```

采集两类时间：

```text
1. Attention / MLA
   input: 16384 tokens/NPU, hidden=7168
   output: T_attn_A3(dtype)

2. Rank-local FFN
   每个 rank 同时负责 8 个 local experts
   tokens/expert sweep:
     512, 1024, 2048, 4096, 8192, 12288
   output: T_ffn_A3(tokens_per_expert, dtype)
```

公开 prefill 参考：CloudMatrix384 arXiv Figure 21(b)，inference prefill，`4K prompt`，
`16K total tokens/NPU`，单位是 `ms/layer`。这里只看量级，不作为训练实测结果。

网络口径：

```text
CM384 内部 MoE dispatch/combine: 走 UB plane，不是 RDMA
CM384 / A3 RDMA plane: 400 Gbps/NPU 单向
A3 服务器 RDMA 聚合: 8 * 400 Gbps = 3.2 Tbps = 400 GB/s 单向
A3 服务器内 D2D: 784 GB/s 双向
```

| stage | with microbatch | without microbatch |
|---|---:|---:|
| Overall | 39.0 ms | 51.4 ms |
| Attention total | 8.3-8.4 ms | 16.5 ms |
| Attn-0 Pre-FA | 2.7 ms | 5.0 ms |
| Attn-1 FA | 3.8-3.9 ms | 7.5 ms |
| Attn-2 Post-FA | 1.8 ms | 4.0 ms |
| Gating | 3.7-4.0 ms | 3.6 ms |
| Dispatch | 7.5-8.0 ms | 10.6 ms |
| MoE / FFN | 7.3-7.8 ms | 14.7 ms |
| Combine | 6.8-10.1 ms | 10.7 ms |

### 2.5 DSV3 单层计算链路

```text
DSV3 MoE layer, training / prefill, one layer
prompt length = 4096
total tokens/NPU = 16384
hidden = 7168
experts = 256, top-k = 8, EP = 32
local experts/rank = 256 / 32 = 8
avg tokens/expert = 16384 * 8 / 256 = 512

                  +-----------------------------------------+
input hidden x --> | MLA / attention, A3 measured profile    |
16384 x 7168      | Q/K/V projection + attention core       |
                  | T_attention_prefill_or_train            |
                  +--------------------+--------------------+
                                      |
                                      v
                  +-----------------------------------------+
                  | Router top-k=8                          |
                  | total expert-token = 16384 * 8          |
                  | avg expert load = 512 tokens            |
                  | hot load = 11x-22x avg                  |
                  +--------------------+--------------------+
                                      |
                                      v
                  +-----------------------------------------+
                  | Dispatch AllToAllV over A3 topology     |
                  | token -> owner rank / owner A3 server   |
                  | hotspot becomes destination incast       |
                  +--------------------+--------------------+
                                      |
                                      v
                  +-----------------------------------------+
                  | Rank-local expert FFN, A3 measured      |
                  | each rank owns 8 local experts          |
                  | FFN: 7168 -> 2*2048 -> 7168 SwiGLU      |
                  | avg case: 512 tokens/expert             |
                  | hot case: 5.6K-11.3K tokens/expert      |
                  | T_expert_FFN(tokens, dtype, A3 profile) |
                  +--------------------+--------------------+
                                      |
                                      v
                  +-----------------------------------------+
                  | Combine AllToAllV                       |
                  | waits for slowest A3 server / rank      |
                  +--------------------+--------------------+
                                      |
                                      v
                            output hidden

Key point:
一层 MoE 等最慢的 A3 服务器 / rank。
热点专家复制后，目标是降低最慢服务器的 FFN 长尾；
新增网络流量只要能被计算 overlap，就可以换来 CCT 下降。
```

## 3. 超节点拓扑建模

网络部分只参考 `/home/chen/workspace/infra/eth-htsim-ocs-eps` 里的拓扑口径，不引用
任何运行性能数值。

建模口径：

```text
A3 server / tray: 8 NPU
in-server NPU D2D bandwidth: 双向 784 GB/s
L0 domain: 64 NPU = 8 台 A3
Group / pod: 512 NPU = 64 台 A3
大规模拓扑背景: 16 pod, 8192 NPU endpoint
network-side link rate: 每条超节点网络链路 400 Gbps
source rails: 每个 endpoint 8 条 400 Gbps 网络 rail
network layers: L0/L1 EPS + L2 OCS
OCS path: 跨 group 走 OCS 候选路径
```

这里的 `400 Gbps` 指网络侧 400GE / EPS / OCS 链路，不是单机内 8 个 NPU 之间的
D2D 卡间互联。单机内 D2D 互联按 A3 公开规格使用双向 `784 GB/s` 口径。
实际评估时从 A3 服务器出发，按作业拿到的服务器集合裁剪拓扑；大规模拓扑只用于
判断碎片化 placement 会跨哪些 L0 domain / group，不把 384 或 8192 endpoint 当成
本 case 的默认规模。

### 3.1 单个 FFN 权重复制数据量

这里的“一个 FFN”指复制一个路由专家的 FFN 权重参数，不是 token
dispatch/combine 数据。

```text
hidden size H = 7168
expert intermediate H' = 2048
路由专家 FFN = gate_proj + up_proj + down_proj

gate_proj: 7168 * 2048
up_proj:   7168 * 2048
down_proj: 2048 * 7168

params/expert = 3 * 7168 * 2048 = 44,040,192 weights
```

| 复制对象 | 数据类型 | 传输数据量 |
|---|---:|---:|
| 单个路由专家 FFN 权重 | BF16 / FP16 | 88,080,384 B = 84 MiB |
| 单个路由专家 FFN 权重 | FP8 / INT8 | 44,040,192 B = 42 MiB |
| 一个 rank 的 8 个路由专家 | BF16 / FP16 | 672 MiB |
| 一个 rank 的 8 个路由专家 | FP8 / INT8 | 336 MiB |

按网络速率估算，复制一个 BF16 路由专家 FFN：

```text
400 Gbps RDMA 单向链路: 84 MiB / 50 GB/s 约 1.76 ms
8-plane 3.2 Tbps 聚合: 84 MiB / 400 GB/s 约 0.22 ms
A3 服务器内 D2D:       84 MiB / 784 GB/s 约 0.11 ms
```

训练时如果远端副本参与反向传播，还要把这个专家的 weight grad 聚合回原始
rank；若按 FP32 grad 计算，单个专家 grad 是 `168 MiB`。

通信域分三类：

```text
L0 域内：
  同一个 64-NPU L0 domain 内通信

L0 域间：
  同一个 group 内、不同 L0 domain 之间通信

Group 域间：
  不同 group / pod 之间通信，需要经过 L2 OCS
```

这套拓扑的作用是回答一个问题：

```text
如果热点专家放在某台 A3 上，
来自其他 source ranks 的 token 会经过哪些 NIC、EPS、OCS 路径，
这些额外网络时间能不能被本地计算掩盖？
```

## 4. 方案逻辑

### 4.1 Baseline：普通 placement

普通 placement 下，每个逻辑专家只在一个物理服务器上执行。DSV3 的
热点专家分布会造成：

```text
热点专家
  -> 热点目的服务器
  -> dispatch incast
  -> FFN compute 长尾
  -> combine 等待最慢服务器
```

如果某个服务器上同时放了多个热点专家，问题会进一步放大。

### 4.2 优化：热点专家冗余

优化不是复制所有专家，而是只处理对关键路径贡献最大的少量热点专家。A3 每个 NPU
有 8 条 400Gbps plane，聚合是 `3.2Tbps` 单向；如果额外传输能被计算 overlap，就
可以用网络换掉最热服务器上的 FFN 长尾。

副本放置用一个简单的 LPT 思想：

```text
1. 按专家热度从大到小排序。
2. 找出当前负载最高的 A3 服务器上的热点专家。
3. 把这个专家复制到当前计算负载最低的 NPU / 服务器。
4. source 端把这个专家的一部分 token 分给副本。
5. 重复少量轮次，直到最慢服务器的 FFN 时间不再明显下降。
```

网络路径也保持简单：

```text
source routing
  -> 在当前候选路径集合里选一条最不拥塞的路径
  -> 优先避开当前最拥塞 link / plane
  -> 不做复杂全局重路由
```

模型语义保持不变：

```text
逻辑专家 id 不变
物理执行位置改变
prefill: combine 仍按原 token 顺序和 expert id 回收
training: backward 梯度按逻辑专家聚合回原专家权重
```

这个方案成立的条件是：

```text
被削掉的热点服务器 FFN 时间
  >
未被 overlap 掩盖的额外网络转发时间
```

在 A3 这种高网络带宽、计算相对更紧的场景下，这个 tradeoff 更容易成立。

### 4.3 结果口径

粗算口径：

```text
专家总数: 256
MoE layers: 58
单个路由专家 FFN 权重: 84 MiB BF16
每层新增专家副本: 4-8 个
说明: 不是 4-8 个不同专家；最热门的专家可能复制多个副本
每层权重复制量: 0.33-0.66 GiB
全模型权重复制量: 19-38 GiB
```

按网络估算：

```text
8-plane 3.2Tbps 理想带宽: 约 0.05-0.10s
考虑路径竞争、调度和分层拷贝: 按约 1s 级别预算
```

热点削峰：

```text
原始热点: 最热专家约 10-20x 平均负载
目标负载: 分流后压到约 3x 平均负载
需要副本: 极热专家通常需要 4-8 个执行位置共同分担
计算长尾收益: 10-20x / 3x，极端情况下 FFN 长尾可接近 6x 加速
新增开销: 热点专家权重预取 + 分流 token 的网络传输
```

所以结果不说“全局仿真提升 XX%”，而说：

```text
复制约 4-8 个热点专家副本后，最热专家从 10-20x 平均负载削到约 3x；
新增开销主要来自热点专家网络侧传输，极端热点层的 FFN 长尾最高可接近 6x 加速。
```

## 5. 面试追问

### Q1：为什么这是 placement 问题？

A：

因为 router 只决定 token 去哪个逻辑专家，但哪个物理服务器承载这个专家由
placement 决定。DSV3 的专家热度已经不均匀，如果热点专家被放到同一批 A3 服务器上，
通信和计算都会集中，最后拖慢整个 MoE 层。

### Q2：为什么要引入 A3 超节点拓扑？

A：

因为多租户资源碎片化下，作业不一定拿到连续服务器。几十卡小集群里看起来合理的
placement，放到超节点里可能跨 L0 domain 或跨 group，通信路径会完全不同。我们要
评估的是热点专家分流后，额外网络路径是否能被 FFN 计算掩盖。

### Q3：为什么可以用网络换计算？

A：

A3 服务器的 D2D 和 400G 网络是相对强项，而 DSV3/MoE 的实际 sustained compute 会
受到低精度 kernel、算子融合、通信 overlap 和热点专家 GEMM 的影响。DeepEP
overlap 后，暴露瓶颈更容易落在热点专家 FFN 计算上。把部分 token 发到副本会
增加网络流量，但如果这部分网络时间被计算 overlap 掩盖，就能降低最慢服务器的
完成时间。

### Q4：为什么不直接说 AllToAll 优化？

A：

AllToAll 优化只能处理通信路径和负载均衡，但不能减少热点专家必须执行的 FFN
计算。这个 case 的关键是通信热点和计算热点叠在同一个服务器上，所以要从专家物理
位置入手。

### Q5：你的贡献怎么讲比较稳？

A：

我不会说自己主导了某企业侧实机实现。更稳的说法是：我做了超节点拓扑建模和瓶颈
分析，把某企业采集到的 DSV3 专家分布、A3 attention/FFN 计算 profile 和
EPS/OCS 网络拓扑放到一个统一模型里，提出热点专家冗余和源端分流方案，并在
某企业 A3 小规模验证中提供技术支持。

## 参考资料

- [Huawei Atlas 800T A3 产品页](https://e.huawei.com/cn/products/computing/ascend/atlas-800t-a3)
- [Atlas 800T A3 Technical Specifications](https://support.huawei.cn/enterprise/en/doc/EDOC1100508910/56869425/technical-specifications)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Serving Large Language Models on Huawei CloudMatrix384](https://arxiv.org/html/2506.12708v1)
- `/home/chen/workspace/infra/eth-htsim-ocs-eps/tests/EXPERIMENTS.md`
- `/home/chen/workspace/infra/eth-htsim-ocs-eps/huawei_dev_docx/require/recognized_requirements.md`
