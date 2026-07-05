
## 零、写在前面：这篇文章在讲什么

我们要聊的这几个东西——[InfiniBand](https://zhida.zhihu.com/search?content_id=272225383&content_type=Article&match_order=1&q=InfiniBand&zhida_source=entity)、RoCE、PFC、ECN、[CBFC](https://zhida.zhihu.com/search?content_id=272225383&content_type=Article&match_order=1&q=CBFC&zhida_source=entity)，以及最新入场的 UEC（[Ultra Ethernet Consortium](https://zhida.zhihu.com/search?content_id=272225383&content_type=Article&match_order=1&q=Ultra+Ethernet+Consortium&zhida_source=entity)）——本质上都是在回答一个问题：

**怎么在数据中心里，把数据从 A 搬到 B，搬得又快又稳又不丢？**

这个问题看似简单，但整个行业围绕它打了二十年架，分裂成了两个阵营，搞出了一堆协议和补丁。现在第三股力量——UEC——入场了，它的宣言是：**别再给以太网打补丁了，我们从传输层重新设计一个。**

今天我们就把这些东西的来龙去脉、彼此关系、内核机制一次性讲透。



## 一、[RDMA](https://zhida.zhihu.com/search?content_id=272225383&content_type=Article&match_order=1&q=RDMA&zhida_source=entity)：一切的起点

要理解后面所有东西，必须先理解 RDMA（Remote Direct Memory Access）。

传统网络通信的路径是这样的：

```text
应用层 → 系统调用 → 内核协议栈(TCP/IP) → 数据拷贝 → 网卡驱动 → 网卡 → 网线
```

这条路径有三个致命开销：**内核态切换、协议栈处理、内存拷贝**。对于吞吐 100Gbps 以上的场景，CPU 光忙着搬数据就累死了，根本没空干正事。

RDMA 的做法是——**全部绕过去：**

```text
应用层 → 网卡（直接 DMA 读写远端内存）
```

没有系统调用，没有内核参与，没有数据拷贝。网卡直接从应用的虚拟地址空间里把数据捞出来，扔到对面机器的内存里。对面的 CPU 甚至不知道这件事发生过。

这就是所谓的 **zero-copy + kernel-bypass + OS-bypass**。

听起来很美，但它有一个绝对的前提条件：

> **网络不能丢包。**

为什么？因为 RDMA 绕过了内核协议栈，也就绕过了 TCP 那套精密的重传机制。RDMA 的传输层（在 IB 协议里叫 Reliable Connection）虽然也有重传，但那个重传机制极其笨重——一旦触发，性能断崖式下跌，甚至直接超时断连。

所以，整个 RDMA 生态的核心矛盾就是：**我需要一个无损网络，但网络天然是有损的。**

三大阵营对此给出了截然不同的答案：IB 说“我从头自己建”，RoCE 说“我在以太网上打补丁”，UEC 说“补丁方向错了，我换个思路”。



## 二、InfiniBand：从第一天就做对了

InfiniBand（简称 IB）是专门为 RDMA 设计的网络体系。注意，不是在已有网络上“改造”出 RDMA，而是 **从物理层到传输层全部重新设计**。

IB 的协议栈长这样：

```text
应用（Verbs API）
    ↓
传输层（RC/UC/UD/XRC）
    ↓
网络层（LID/GRH 路由）
    ↓
链路层（信用流控 + VL）
    ↓
物理层（专用线缆和交换机）
```

它和 TCP/IP 没有任何关系。自己的编址（LID），自己的路由（子网管理器），自己的传输协议，自己的流控。

### IB 怎么做到无损？——Credit-Based Flow Control（CBFC）

这是整篇文章最关键的概念之一。

IB 链路层的流控机制是 **基于信用（Credit）** 的。原理如下：

1. 接收方初始化时告诉发送方：**“我有 N 个 buffer 空位，你最多发 N 个包。”**
2. 发送方每发一个包，信用值减 1。
3. 信用值减到 0，**发送方自己停下来**，不需要任何人喊它停。
4. 接收方每消化一个包，释放一个信用，通知发送方。
5. 发送方收到新信用，继续发。

```text
发送方                          接收方
  |                              |
  |  ← Credit = 8 ────────────── │   "我能收8个"
  |                              |
  │──── Packet 1 ──────────────→ │   Credit: 8→7
  │──── Packet 2 ──────────────→ │   Credit: 7→6
  │──── ...                      │
  │──── Packet 8 ──────────────→ │   Credit: 1→0
  |                              |
  |  （信用为0，自己停下）         │
  |                              │   处理完 Packet 1-3
  |  ← Credit += 3 ───────────── │   "又腾出3个位置了"
  |                              |
  │──── Packet 9 ──────────────→ │
  │──── Packet 10 ─────────────→ │
  │──── Packet 11 ─────────────→ │
```

**这个机制的精妙之处在于——它是预防式的，而不是反应式的。**

接收方的 buffer 永远不会溢出，因为发送方不可能发出超过接收方容量的数据。不存在“快满了才喊停”这种后知后觉的操作。就像高速公路收费站前面的匝道信号灯——你不是等收费站堵了才拦车，而是根据收费站的处理能力，提前控制匝道放行的车辆数。

而且 CBFC 是 **逐链路、逐队列** 的。Switch 端口 A 的队列 3 没信用了，完全不影响端口 A 的队列 0-2、4-7，也不影响端口 B 的任何队列。**粒度极细，隔离性极强。**

### IB 还有什么好东西？

**Virtual Lanes（VL）**：IB 在链路层支持最多 16 条虚拟通道（VL0-VL15），每条 VL 有独立的 buffer 和信用计数。不同优先级的流量走不同 VL，天然隔离，不存在队头阻塞。VL15 专门留给管理报文，保证管理通道永远不被数据流量挤占。

**子网管理器（SM）**：集中式路由计算，全局视角优化路径，避免环路和拥塞热点。

**总结：IB 从芯片到协议到管理面，一切都是为 RDMA 定制的。无损不是靠补丁实现的，而是骨子里写好的。**



## 三、以太网的野心：RoCE

IB 好是好，但有个致命问题——**贵，而且封闭。**

专用交换机、专用线缆、专用网卡、专用管理软件。数据中心已经铺了一整套以太网基础设施，你让我再搞一套 IB 网络？两张网的运维成本、布线成本、人员成本，这笔账算不过来。

于是行业里出现了一个想法：**能不能在以太网上跑 RDMA？**

这就是 RoCE——RDMA over Converged Ethernet。

### RoCE v1

第一版很粗糙：把 IB 的传输层直接封装在以太网帧里。

```text
[Ethernet Header] [IB Transport Header] [Payload] [ICRC]
```

问题：以太网帧不能路由，只能在同一个二层域内工作。跨子网？做不到。

### RoCE v2

第二版聪明了一点：在以太网帧和 IB 传输层之间加了 UDP/IP 头。

```text
[Ethernet Header] [IP Header] [UDP Header] [IB Transport Header] [Payload] [ICRC]
```

现在可以路由了，可以跨子网了。UDP 目标端口固定用 4791。

看起来问题解决了？

**远远没有。**

RoCE v2 只是解决了“怎么把 RDMA 报文装进以太网帧”这个封装问题。但以太网骨子里的根本矛盾没有解决：

> **以太网是有损的。**

以太网的设计哲学从第一天起就是 **best-effort delivery**——尽力而为，丢了拉倒。交换机 buffer 满了？尾丢弃（tail drop），没商量。这对 TCP 来说完全没问题，因为 TCP 有精密的重传和拥塞控制。但对 RDMA 来说，这是灭顶之灾。

所以，要在以太网上跑 RDMA，你必须先把以太网 **改造成无损网络**。

怎么改？两件武器：**PFC 和 ECN。**



## 四、PFC：以太网的紧急刹车

### 4.1 从 PAUSE 帧说起

以太网其实很早就有流控机制——IEEE 802.3x PAUSE 帧。接收方 buffer 快满了，发一个 PAUSE 帧给发送方，发送方停止发送一段时间（由 PAUSE 帧里的 quanta 值决定）。

问题是，802.3x PAUSE 是 **全端口暂停**。一发 PAUSE，这个端口上所有流量全部停下。你的 RDMA 流量堵了，你的管理流量、存储流量、心跳包全都跟着停——这叫同归于尽。

### 4.2 PFC 的改进

PFC（Priority-based Flow Control，IEEE 802.1Qbb）在 PAUSE 帧的基础上加了一个维度：**优先级**。

以太网帧的 VLAN Tag 里有 3 bit 的 PCP（Priority Code Point）字段，可以表示 0-7 共 8 个优先级（也叫 CoS，Class of Service）。PFC 可以 **针对某一个优先级单独发 PAUSE**，不影响其他优先级。

PAUSE 帧格式可以理解成下面这样：

| Priority | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 动作 | 继续发 | 继续发 | 继续发 | 暂停 | 继续发 | 继续发 | 继续发 | 继续发 |
| quanta | 0 | 0 | 0 | 800 | 0 | 0 | 0 | 0 |

比如，RDMA 流量走 priority 3，管理流量走 priority 7。当 priority 3 的 buffer 快满了，PFC 只暂停 priority 3，priority 7 的管理流量照走不误。

### 4.3 PFC 的工作阈值

PFC 是 **反应式** 的。交换机对每个优先级的入队列设两个水位线：

- **XOFF 阈值**：buffer 用量超过这个值，发送 PFC PAUSE（暂停）
- **XON 阈值**：buffer 用量降到这个值以下，发送 PFC UNPAUSE（恢复）

```text
Buffer 使用量
  ▲
  │ ████████████████████ ←── Buffer 上限（溢出 = 丢包）
  │
  │ ──────────────────── ←── XOFF 阈值（发 PAUSE）
  │ ████████████████
  │ ████████████████
  │ ──────────────────── ←── XON 阈值（发 UNPAUSE）
  │ ██████████
  │ ██████████
  └─────────────────────→ 时间
```

XOFF 和 buffer 上限之间的空间叫 **headroom**——这是留给“在途报文”的。因为 PAUSE 帧发出去后，对方不会立刻停下来（光在线缆里传播需要时间，对方已经发出的包也需要时间到达），这段时间里还会有报文涌入。headroom 必须够大，大到能吃下这些在途报文，否则还是会丢包。

**headroom 的计算公式（简化版）：**

$$headroom = 2 \times D_{propagation} \times LineRate + MTU_{max} \times N_{ports}$$

其中 $D_{propagation}$ 是链路传播延迟。线缆越长，headroom 需要越大。这就是为什么 300 米的光缆比 3 米的 DAC 铜缆需要更多的交换机 buffer。

### 4.4 PFC 的致命问题

#### 问题一：PFC Storm（PFC 风暴）

PFC 的 PAUSE 是 **逐跳传播** 的。考虑一条链路：

```text
Server A → Switch1 → Switch2 → Switch3 → Server B（慢节点）
```

Server B 处理慢，Switch3 的 buffer 满了，Switch3 向 Switch2 发 PAUSE。Switch2 对应优先级的队列开始积压，buffer 也满了，Switch2 向 Switch1 发 PAUSE。Switch1 的 buffer 满了，Switch1 向 Server A 发 PAUSE。

**整条链路上同一优先级的所有流量全部冻结。** 而 Switch1、Switch2 上被暂停的流量里，可能有完全跟 Server B 无关的——它们只是恰好走同一优先级、恰好经过同一条链路。

这就是所谓的 **HOL Blocking（Head-of-Line Blocking）**——队头阻塞。PFC 只有 8 个优先级，所有 RDMA 流量共享一个优先级，一个害群之马能把整个优先级的所有流量拖下水。

更可怕的是，如果网络拓扑存在环路，或者某个节点的网卡固件有 bug 持续发 PAUSE，PFC Storm 可以 **永久冻结** 整个 fabric 的某个优先级。

#### 问题二：不公平

PFC 是二进制的——要么全速发，要么完全停。没有中间状态。如果 10 条流共享一个出口，其中 1 条流导致了拥塞，PFC 一发 PAUSE，10 条流全停。那条行为良好的小流量和那条作恶的大象流受到完全相同的惩罚。

#### 问题三：死锁

在特定拓扑下（特别是有环路的 Clos 拓扑中存在上下行反转的情况），PFC 可能导致循环依赖——A 等 B 释放，B 等 C 释放，C 等 A 释放。**永久死锁，只能靠重启恢复。**

**一句话：PFC 把以太网的“丢包”问题变成了“停顿”问题。丢包至少是局部的、可恢复的；停顿可能是全局的、持久的。这是用一种病换另一种病。**



## 五、ECN：端到端的预警系统

PFC 的问题根源在于——它是 **链路层的被动反应**。buffer 都快炸了你才动手，太晚了。

我们需要一个 **提前量更大、粒度更细、端到端** 的拥塞信号。这就是 ECN。

### 5.1 ECN 的标记机制

IP 头有一个 2-bit 的 ECN 字段（在 DSCP/TOS 字节的最低 2 位）：

| 值 | 含义 |
| ----- | ----- |
| 00 | Not ECN-Capable（不支持 ECN） |
| 01 | ECT(1)（支持 ECN，端点声明） |
| 10 | ECT(0)（支持 ECN，端点声明） |
| 11 | CE（Congestion Experienced，中间设备标记） |

工作流程：

1. **发送方**在 IP 包里设置 ECN = `01` 或 `10`，表示“我支持 ECN，如果路上堵了请告诉我”。
2. **交换机**发现某个队列的平均长度超过阈值（注意：是 **平均** 长度，不是瞬时长度），就把经过的包的 ECN 字段改为 `11`（CE）。
3. **接收方**收到 CE 标记的包，生成一个 **CNP（Congestion Notification Packet）** 发回给发送方。
4. **发送方**收到 CNP，**降低发送速率**。

```text
发送方                  Switch（拥塞点）               接收方
  │                         │                           │
  │── [ECN=01] Packet ────→ │                           │
  │                         │ buffer 超阈值              │
  │                         │ 标记 ECN=11 (CE)           │
  │                         │── [ECN=11] Packet ──────→ │
  │                         │                           │
  │                         │              检测到 CE 标记│
  │                         │                           │
  │← ──── CNP（减速！）──── ← ───────────────────────────│
  │                         │                           │
  │  降低发送速率            │                           │
```

### 5.2 ECN 的阈值设置——WRED

交换机上通常用 **WRED（Weighted Random Early Detection）** 来决定是否标记 CE：

- 队列平均长度 < **K\_min**：不标记
- 队列平均长度在 **K\_min** 和 **K\_max** 之间：按概率标记（越接近 K\_max，概率越大）
- 队列平均长度 > **K\_max**：100% 标记

```text
标记概率
  ▲
  │          ╱ ────── 100%
  │         ╱
  │        ╱
  │       ╱
  │      ╱
  │     ╱
  └────┬──────┬──────→ 队列平均长度
     K_min  K_max
```

**K\_min 要远低于 PFC 的 XOFF 阈值。** 这是核心设计意图——ECN 要在 PFC 触发之前就开始工作，通过提前降速，避免 PFC 被触发。PFC 只是最后的安全网。

### 5.3 [DCQCN](https://zhida.zhihu.com/search?content_id=272225383&content_type=Article&match_order=1&q=DCQCN&zhida_source=entity)：RoCE 世界的拥塞控制算法

ECN 只是一个信号机制，它不规定发送方收到 CNP 后具体怎么降速、之后怎么恢复。在 RoCE 场景中，业界广泛使用的是 **DCQCN（Data Center QCN）** 算法。

DCQCN 的速率调整逻辑：

```text
收到 CNP → 速率立刻乘以一个因子降下来（快速降速）
    ↓
一段时间没收到 CNP → 速率缓慢线性恢复（慢慢加速）
    ↓
又收到 CNP → 再次快速降速
    ↓
如此往复，最终在一个合理的速率上振荡收敛
```

这和 TCP 的 AIMD（加性增、乘性减）思路类似，但参数针对数据中心的微秒级延迟做了优化。

### 5.4 ECN 的局限

**ECN 不保证无损。** 它只是一个建议，发送方降速需要一个 RTT 的反应时间，这段时间里如果 buffer 确实满了，包该丢还是得丢。所以在 RoCE 网络中：

> ECN 负责常态拥塞控制（90% 的情况），PFC 负责兜底（10% 的极端 burst）。

理想状态下 PFC 永远不被触发。一旦 PFC 频繁触发，说明你的 ECN 阈值设得不对，或者网络本身有结构性问题。



## 六、把它们放在一起：RoCE 网络的完整流控架构

一个配置正确的 RoCE v2 网络，流控体系是这样分层的：

| 层级 | 机制 | 位置 | 反应速度 | 控制粒度 | 丢包风险 | 作用 |
|---|---|---|---|---|---|---|
| 第一层 | DCQCN/ECN | 端到端 | 最慢（RTT 级） | 每条流 | 无 | 只降速，常态拥塞控制 |
| 第二层 | PFC | 逐跳 | 较快（链路级） | 每优先级 | 无 | 控制不住时暂停发送 |
| 第三层 | Buffer 吸收 | 交换机本地 | 最快（纳秒级） | N/A | 无 | 吸收短时 burst，但 buffer 有限 |
| 第四层 | 丢包 | 交换机本地 | —— | —— | 有 | buffer 满后的灾难结果 |

各阈值的大小关系**必须**满足：

$$ECN_{Kmin} < ECN_{Kmax} < PFC_{XOFF} < PFC_{XOFF} + Headroom < Buffer_{total}$$

如果这个顺序搞反了——比如 ECN 的 K\_max 比 PFC 的 XOFF 还高——那 ECN 形同虚设，每次都是 PFC 先触发，你就回到了纯 PFC 那个噩梦模式。



## 七、CBFC vs PFC：正规军和游击队的本质差异

现在你应该能深刻理解，为什么说 CBFC 是正规军、PFC 是游击队了。把它们的机制放在一起对比：

| 维度 | CBFC（IB） | PFC（以太网） |
| ----- | ----- | ----- |
| 控制时机 | 事前预防（没信用不准发） | 事后反应（快满了才喊停） |
| 控制方式 | 发送方自律 | 接收方强制暂停发送方 |
| 粒度 | 每条虚拟通道（VL）独立信用 | 8 个优先级共享 |
| 队头阻塞 | 不存在（VL 间完全隔离） | 严重（同优先级所有流受影响） |
| 风暴/死锁 | 不存在 | PFC Storm、死锁是真实风险 |
| 在途报文浪费 | 没有（不会发超额数据） | 有（PAUSE 后在途报文仍需 buffer 吸收） |
| 对 buffer 的需求 | 小（精确控制） | 大（需要 headroom 兜底） |
| 实现复杂度 | 高（需端到端信用管理） | 低（交换机加个阈值） |

**核心区别用一句话说：CBFC 是“你能收多少我就发多少”，PFC 是“你收不下了你再告诉我停”。前者永远不会出问题，后者总有手忙脚乱的时候。**

以太网为什么不用 CBFC？因为以太网的设计基因是 **无状态转发**。每个包独立路由，交换机不记住任何关于“流”的信息。你让交换机对每个端口的每个优先级维护信用计数器、收发信用更新报文——这从根本上违背了以太网的设计哲学。以太网交换机追求的是高端口密度、低成本、简单快速，不是精细流控。

所以以太网只能用 PFC 这种 **简单粗暴的方案** 来模拟无损。PFC 不需要维护状态，不需要信用交换，只需要一个 buffer 水位线和一个 PAUSE 帧——简单到任何交换机芯片都能实现。

但简单的代价就是粗糙。



## 八、UEC：第三条路——不再执着于“无损”

### 8.1 为什么需要 UEC？

到这里你应该能感受到一个根本性的矛盾：

- **IB**：无损做得好，但贵、封闭、生态小。
- **RoCE**：借以太网的基础设施，但用 PFC+ECN 模拟无损是脆弱的，PFC Storm、死锁、配置复杂度像定时炸弹。

行业需要的是什么？**以太网的成本和生态 + IB 级别的性能和可靠性。** RoCE 试了，但它的方式是“在以太网上硬搞无损”，结果搞出了一堆补丁的补丁。

UEC（Ultra Ethernet Consortium）的思路完全不同。UEC 是 Linux Foundation 下的一个行业联盟，成员包括 Meta、微软、AMD、Intel、Broadcom、Cisco 等巨头。2025 年发布了 UE Specification v1.0。它的核心宣言是：

> **别再给以太网打无损补丁了。我们设计一个原生能容忍丢包的传输层，运行在普通的 best-effort 以太网上。**

这句话的分量需要仔细品：RoCE 的整个设计前提是“网络必须无损”，所以才需要 PFC 兜底。UEC 说——**如果传输层自己就能优雅地处理丢包呢？那 PFC 不就不需要了吗？**

### 8.2 UET：传输层的彻底重构

UEC 定义的传输层叫 **UET（Ultra Ethernet Transport）**。UET 的设计目标写得很明确：**“address the shortcomings of RoCEv2, specifically its semantics, transport layer, wire operations, implementation complexities, and scale limits”**——针对 RoCE v2 的语义、传输层、线上操作、实现复杂性和规模上限，逐个解决。

UET 的关键设计决策：

#### 决策一：基于窗口，而非基于速率

RoCE/DCQCN 是 **基于速率（rate-based）** 的拥塞控制——收到 CNP 就降速率，没收到就慢慢升。这个方式有一个致命缺陷：**在 best-effort 网络中，“没收到反馈”可能意味着包被丢了，而不是网络畅通。**

DCQCN 会把“没收到 CNP”理解为“网络不拥塞”，然后加速发送。在有 PFC 兜底的无损网络中这没问题（最差情况 PFC 暂停你）。但在 best-effort 网络中，没有 PFC 兜底，你越加速丢得越多，越丢越没反馈，越没反馈越加速——正反馈死亡螺旋。

UET 选择 **基于窗口（window-based）** 的拥塞控制。窗口的核心特性是 **packet clocking（包时钟）**：发出去的包必须被确认（ACK、Credit、或者 NACK）才能发新的。如果网络严重拥塞，确认回不来，发送方自然就停了——不需要任何人告诉它停，它自己因为窗口用尽而停。

**这和 IB 的 CBFC 在哲学上是一致的——都是“没有许可就不发”，只不过 CBFC 工作在链路层，UET 的窗口工作在传输层（端到端）。**

#### 决策二：三层拥塞控制架构

UET 不是只有一个拥塞控制算法，而是定义了三个互补的机制：

| 机制 | 是否必须 | 核心信号 | 主要作用 |
|---|---|---|---|
| NSCC（Network-Signal Congestion Control） | 必须实现 | ECN 信号 + 发送窗口 | 应对网络核心拥塞，例如 oversubscription 和路径不均衡 |
| RCCC（Receiver-Credit Congestion Control） | 可选实现 | 接收方给发送方分配信用额度 | 专门处理 incast：接收方根据当前有多少 source 同时发送，动态分配信用 |
| TFC（Transport Flow Control） | 可选实现 | 点对点信用流控 | 用于 buffer 有限、不允许丢包的点对点场景 |

注意 **RCCC** 这个设计——它本质上就是 **把 IB 的 CBFC 思想搬到了传输层**。接收方根据自己入口链路的容量和当前活跃的发送方数量，按比例给每个发送方分配信用。100 个源同时发数据过来（incast），接收方把信用平均分成 100 份，每个源只能发 1⁄100 的量。这完美解决了 PFC 永远解决不了的 incast 公平性问题。

更精妙的是，UEC 规范建议：**在最后一跳（last-hop，即交换机到目的网卡的链路）关闭 ECN，让 RCCC 接管 incast 控制。** 因为最后一跳的拥塞本质上就是 incast，RCCC 比 ECN 处理得更精确；而 ECN 信号留给 NSCC 处理网络核心的拥塞。各司其职，互不干扰。

#### 决策三：Packet Trimming——丢包的艺术

这可能是 UEC 最有想象力的设计。

传统以太网交换机处理拥塞的方式是 **尾丢弃（tail drop）**——buffer 满了，新来的包直接扔掉，无声无息。发送方不知道包被丢了，接收方也不知道，只能等超时重传。在微秒级延迟的数据中心里，超时定时器通常是毫秒级的——等一次超时的时间足够传几千个包了。

Packet Trimming 的做法是：**交换机 buffer 快满的时候，不是把包整个扔掉，而是把 payload 砍掉，只保留包头（至少保留到 UET 的 PDS Request header），然后修改 DSCP 标记为 DSCP\_TRIMMED，继续把这个“瘦身版”的包转发给目的端。**

```text
正常包：[Header: 80B] [Payload: 4016B] = 4096B    → 正常转发
Trimmed包：[Header: 80B] [空]          ≈ 80B       → payload 砍掉，包头继续转发
```

这意味着什么？

1. **接收方立刻知道哪个包丢了**——因为 trimmed 包头到了，里面有序列号。不需要等超时，立刻发 NACK 请求重传。恢复时间从毫秒级降到微秒级。
2. **接收方知道网络在哪条路径上拥塞了**——trimmed 包的 entropy value（路径标识）告诉接收方是哪条路出了问题，可以反馈给发送方，让发送方避开这条路径。
3. **Trimmed 包走高优先级队列**——因为只有几十字节，极小，不占带宽。规范要求 trimmed 包进入单独的 traffic class，通常配置为高优先级，所以它能绕过正在排队的数据包，快速到达目的端。
4. **天然防止拥塞崩溃**——trimmed 包走单独队列、有带宽上限。即使极端 incast 下所有包都被 trimmed，trimmed 包头最多占 50% 带宽（WDRR 调度），不会像 Cut Payload 方案那样出现“网络里全是包头没有数据”的崩溃场景。

```text
传统丢包：
  发送方 ──[数据包]──→ Switch ──→ [丢弃，静默] ──→ 接收方（什么都不知道）

  ......等待超时（毫秒级）......

  发送方 ──[重传]──→

Packet Trimming：
  发送方 ──[数据包]──→ Switch ──→ [砍payload，改DSCP] ──→ [Trimmed包头] ──→ 接收方
                                                                             │
                                                         立刻知道丢了什么     │
                                                         立刻 NACK 请求重传   │
                            ←──────────────── [NACK] ────────────────────────│
  发送方 ──[重传，换路径]──→                                （微秒级恢复）
```

**这是一个把“丢包”从灾难变成信号的设计。** 传统网络里丢包是黑洞——包消失了，两端都不知道。Packet Trimming 把丢包变成了一封快递短信：“你的包裹在某某路口被拦下了，请重新发送。”

### 8.3 UEC 对 PFC 和 CBFC 的态度

这是最有意思的部分。UEC 规范白纸黑字写道：

> **“UET-CC is designed for operation in a best-effort network and assumes that PFC is not enabled for end-to-end UET traffic on switch-to-switch links in the network. Enabling PFC on such links is likely to reduce performance, as it will violate the latency assumptions of the CC algorithms. PFC SHOULD NOT be used anywhere in a best-effort network.”**

翻译成人话：**在跑 UET 的网络上，不要开 PFC。开了反而会坏事。**

为什么？因为 PFC 的逐跳暂停会引入不可预测的延迟，而 UET 的拥塞控制算法（NSCC、RCCC）是基于 RTT 估算来调整窗口大小的。PFC 一暂停，RTT 突然飙高，算法的假设全部失效。这就像你在做菜，配方要求精确温控，结果有人时不时关一下煤气——你的温度曲线全乱了。

**对 CBFC 也是同样的态度：** UEC 在链路层定义了自己的 CBFC 规范（支持多达 32 个 VC，比 IB 的 16 个 VL 还多），但明确说 CBFC 也不应该用在 switch-to-switch 的 UET 流量上。CBFC 的定位是 **可选的链路层增强**——用在特定的、对丢包零容忍的点到点场景，比如 NIC-to-Switch 的管理通道。

这意味着 UEC 做了一个根本性的范式转换：

```text
RoCE 的世界观：
  "网络必须无损" → PFC 兜底 → ECN 尽量别让 PFC 触发 → 传输层假设不丢包

UEC 的世界观：
  "网络就是会丢包" → 传输层自己优雅地处理丢包 → Packet Trimming 加速丢包感知
  → NSCC+RCCC 精细控速 → PFC/CBFC 变成可选的、极少数场景才用的东西
```

### 8.4 UEC 的 CBFC：比 IB 更精细

虽然 UEC 不主张在 UET 流量上用 CBFC，但它确实在链路层定义了一套 CBFC 规范。值得注意的是，UEC 的 CBFC 比 IB 原生的 CBFC 和以太网的 PFC 都更精细：

| 维度 | PFC | IB CBFC | UEC CBFC |
| ----- | ----- | ----- | ----- |
| 虚拟通道数 | 8（CoS 0-7） | 16（VL0-VL15） | 最多 32 个 VC |
| 信用粒度 | 无（二进制开关） | 固定 | 可配置 CreditSize（32B-2048B） |
| 协商机制 | 无 | SM 分配 | LLDP TLV 动态协商 |
| 与 PFC 共存 | N/A | N/A | 支持同时运行 |
| 线缆长度敏感 | 非常敏感（headroom 计算） | 中等 | 不敏感（低估只是降吞吐，不丢包） |
| 信用泄漏恢复 | N/A | 需重置 | 循环计数器自动同步 |

UEC CBFC 有一个特别值得注意的优势：**对线缆长度不敏感。** PFC 的 headroom 计算必须精确考虑线缆传播延迟，算少了就丢包。而 CBFC 如果低估了可用信用，最差结果只是链路利用率不满，不会丢包。这在实际部署中意味着更低的配置风险和更少的半夜被叫醒。

UEC CBFC 还使用 **循环计数器（cyclic counter）** 来跟踪 credits consumed 和 credits freed，天然能从信用泄漏中自动恢复——下一次同步报文来了，计数器差值自动纠正。不需要像某些实现那样检测到泄漏后重置整条链路。

### 8.5 多路径喷洒：另一个维度的创新

RoCE 的世界里，一条 QP 连接（flow）的所有包走同一条路径（通过 ECMP hash 决定），这导致大象流容易把某条路径打满，而其他路径闲着。

UET 从设计之初就把 **逐包多路径喷洒（per-packet spraying）** 写进了协议：

```text
RoCE：
  Flow A 的所有包 ──────→ Path 1 ──────→ 目的端
  Flow B 的所有包 ──────→ Path 2 ──────→ 目的端
  （如果 Flow A 是大象流，Path 1 爆了，Path 3/4/5 还闲着）

UET：
  Flow A 的 Packet 1 ──→ Path 1 ──→ ┐
  Flow A 的 Packet 2 ──→ Path 3 ──→ ├→ 目的端（乱序到达，传输层重排）
  Flow A 的 Packet 3 ──→ Path 2 ──→ ┘
  （所有路径均匀负载）
```

UET 使用 256 个 entropy value（可配置）来做路径选择。发送方轮流使用不同的 entropy，交换机根据 entropy hash 到不同路径。如果某条路径的包被 ECN 标记或被 trimmed，发送方知道这条路径拥塞了，可以暂时减少向这条路径发包——这就是 **path-aware multipath spraying**。

这意味着 UET 的拥塞控制不仅控制“发多快”，还控制“往哪发”——空间和时间两个维度同时优化。

当然，逐包喷洒带来一个副作用：**乱序到达。** UET 对此的处理是：传输层原生支持乱序（定义了 RUD——Reliable Unordered Delivery 模式），大块数据传输允许乱序到达后在目的端重组，只在消息级别保序。对 AI 集合通信（XCCL 操作）来说，这完全够用——AllReduce 只关心最终结果对不对，不关心每个 chunk 的到达顺序。

### 8.6 三个 Profile：不是一刀切

UET 定义了三个配置文件，适应不同场景：

| Profile | 目标场景 | 必须支持 | 关键特性 |
| ----- | ----- | ----- | ----- |
| AI Base | 最低成本的 AI 训练 | NSCC、基本 RDMA | 最精简实现 |
| AI Full | 完整 AI 能力 | AI Base + Deferrable Send + Exact Match | 硬件可卸载 XCCL 操作，省 1 个 RTT |
| HPC | 高性能计算 | 几乎全部功能 | 原子操作、完整 RDMA 语义 |

这意味着一个做 AI 加速卡的芯片厂商可以只实现 AI Base profile，用最小的硬件代价获得 UET 的核心收益（多路径、Packet Trimming、窗口拥塞控制）。不需要像 IB 那样实现完整的 Verbs 语义才能入场。

### 8.7 规模：为 AI 超级集群而生

UET 的规模目标令人印象深刻：

- **网络规模**：100K - 256K 以太网端口
- **端点数量**：支持百万级 NIC 端口，寻址十亿级进程
- **端口速率**：800G+
- **目标单向延迟**：2 - 10 μs
- **平均网络利用率**：高达 85%

256K 端口的以太网集群，跑 800G 端口速率，85% 利用率——这是当前 RoCE 网络在配置和运维上极难达到的目标。UET 通过原生多路径 + 精细拥塞控制 + Packet Trimming，试图让这个目标变得可行。



## 九、全景对比：四种方案的灵魂拷问

| 维度 | InfiniBand | RoCE v2 | UEC/UET | 纯 TCP/IP |
| ----- | ----- | ----- | ----- | ----- |
| 对丢包的态度 | 不允许（CBFC 预防） | 不允许（PFC 兜底） | 允许并优雅处理 | 允许（TCP 重传） |
| 流控机制 | 链路层 CBFC | 链路层 PFC + 端到端 ECN/DCQCN | 端到端窗口(NSCC/RCCC) + 可选链路层 CBFC | 端到端 TCP 窗口 |
| 多路径 | 有限（子网管理器分配） | 几乎没有（per-flow ECMP） | 原生逐包喷洒 | MPTCP（有限） |
| 拥塞信号 | 隐式（信用耗尽） | ECN + CNP | ECN + Packet Trimming | 丢包 + ECN |
| Incast 处理 | CBFC 天然限速 | PFC 暂停（粗暴） | RCCC 接收方精确信用分配 | TCP 超时重传（慢） |
| PFC 依赖 | 不需要 | 强依赖 | 明确不建议使用 | 不需要 |
| 交换机要求 | 专用 IB 交换机 | 支持 PFC/ECN 的以太网交换机 | 普通以太网交换机即可（trimming 可选） | 任意交换机 |
| 规模上限 | ~48K 端口（SM 瓶颈） | ~100K（PFC 域管理困难） | 256K 端口 | 无限制 |
| 成本 | 最高 | 中等 | 低（复用以太网基础设施） | 最低 |
| 延迟 | 最低（~1μs） | 低（~2-5μs，无 PFC Storm 时） | 低（~2-10μs） | 高（~50-100μs） |



## 十、实战中的关键配置和典型翻车

### 10.1 交换机端 PFC 配置（RoCE 场景）

以 Mellanox Spectrum 系列交换机为例（Cumulus Linux）：

```text
# 启用 priority 3 的 PFC
sudo mlnx_qos -i swp1 --pfc 0,0,0,1,0,0,0,0

# 查看 PFC 计数器
ethtool -S swp1 | grep pfc
#   rx_pfc3_pause: 12847     ← 收到了 12847 个 PAUSE 帧（上游在喊停）
#   tx_pfc3_pause: 0         ← 没发过 PAUSE（说明本端不拥塞）
```

如果 `rx_pfc_pause` 计数器疯涨，说明你的下游有拥塞，而且 PFC 已经在逐跳回压了。这时候该去查的不是这台交换机，而是 **下游的下游**。

### 10.2 网卡端 ECN 配置

```text
# 查看网卡 ECN 配置
mlnx_qos -i mlx5_0 --ecn show

# 设置 ECN 参数：对 priority 3 启用 ECN
mlnx_qos -i mlx5_0 --ecn 0,0,0,1,0,0,0,0

# 确认 DCQCN 参数
sysctl net.ipv4.tcp_ecn
```

### 10.3 RoCE v2 的 GID 配置——最隐蔽的坑

RoCE v2 基于 UDP/IP，网卡需要知道自己用哪个 IP 地址和 GID。每块网卡有一个 GID 表：

```text
$ show_gids
DEV     PORT  INDEX  GID                                     IPv4            VER   DEV
mlx5_0  1     0      fe80:0000:0000:0000:0a2c:29ff:fe1a:3b4c                 v1    enp1s0f0
mlx5_0  1     1      0000:0000:0000:0000:0000:ffff:c0a8:010a 192.168.1.10    v2    enp1s0f0
mlx5_0  1     2      fe80:0000:0000:0000:0a2c:29ff:fe1a:3b4c                 v2    enp1s0f0
```

**RoCE v2 必须使用 VER=v2 的 GID。** 很多初学者选了 index 0（v1），结果跨子网通信直接失败。报错信息是毫无帮助的 `Transport retry count exceeded`——你看到这个报错，第一反应应该是去检查 GID index。

在 NCCL 里对应的环境变量：

```text
export NCCL_IB_GID_INDEX=1    # 对应上面 v2 且有 IPv4 的那条
```

### 10.4 典型翻车场景汇总

| 场景 | 症状 | 原因 | 解法 |
| ----- | ----- | ----- | ----- |
| 只配 PFC 不配 ECN | 小规模正常，一上规模全卡 | PFC Storm | 加上 ECN，设好阈值 |
| ECN 阈值 > PFC 阈值 | ECN 形同虚设，PFC 疯狂触发 | 阈值顺序错误 | 确保 ECN_Kmax < PFC_XOFF |
| GID index 选错 | 同子网通，跨子网不通 | 用了 v1 的 GID | 用 show_gids 确认 v2 index |
| headroom 算少了 | PFC 开了还丢包 | 在途报文超过 headroom | 按线缆长度重算 headroom |
| 在 IB 网络上配 PFC | 多此一举，可能引入问题 | IB 用 CBFC 不需要 PFC | 删掉 PFC 配置 |
| PFC 双端不匹配 | 一端 PFC 一端没开，丢包 | PFC 需要链路两端都启用 | 确保两端 + 中间交换机全启用 |
| UET 网络上开了 PFC | 延迟抖动大，CC 算法失效 | PFC 暂停破坏 RTT 估算 | 关掉 PFC，信任 UET-CC |



## 十一、收尾：一张全景图

RDMA 的目标始终没有变：零拷贝、内核旁路、低延迟。真正变化的是底层网络如何提供可靠的高性能数据传输。

| 路线 | 基本定位 | 核心机制 | 网络观 |
|---|---|---|---|
| InfiniBand | 原生无损网络 | CBFC 信用流控 | 链路层预防，天然无损 |
| RoCE v2 | 以太网 + 无损补丁 | PFC + ECN | 链路层暂停 + 端到端预警，模拟无损 |
| UEC / UET | 以太网 + 新传输层 | NSCC + RCCC + Packet Trimming | 端到端智能，允许丢包并快速恢复 |

### 时间线演进

| 时期 | 核心理念 | 代表技术 |
| ----- | ----- | ----- |
| 2000年代 | “我自己建一套” | InfiniBand |
| 2010年代 | “借你的路改一改” | RoCE v2 |
| 2025年+ | “你的路我重铺一层” | UEC / UET |



**技术对比说明**

- **InfiniBand**：原生无损网络，基于链路层信用流控（CBFC），从设计之初就保证无丢包，但需要专用网络设备。
- **RoCE v2**：在标准以太网上通过 PFC（优先级流控）和 ECN（显式拥塞通知）模拟无损环境，实现成本较低，但存在网络死锁和 PFC 风暴等风险。
- **UEC / UET**：面向超以太网联盟的新一代传输协议，采用端到端智能拥塞控制（如 NSCC、RCCC）和数据包修剪（Pkt Trimming），允许丢包并通过上层机制保障可靠性，旨在兼顾高性能与通用性。

## 十二、最后的最后

回头看这二十年的技术演进，有一条清晰的思想脉络：

**IB 说：网络不能丢包，我从底层保证不丢。** ——做对了，但太贵。

**RoCE 说：网络不能丢包，我在以太网上加锁保证不丢。** ——思路对了一半，但锁（PFC）本身成了最大的问题。

**UEC 说：网络就是会丢包。与其耗尽精力去消灭丢包，不如设计一个传输层让丢包不再可怕。** ——这是范式转换。

UEC 的 Packet Trimming 把“丢包”从黑洞变成了信号，NSCC+RCCC 用窗口+信用实现了精细控速，多路径喷洒把单条路径的拥塞消解到整个 fabric。这三板斧加在一起，让 PFC 这个曾经不可或缺的补丁变成了可选项。

这不是说 PFC 和 CBFC 没用了——在 NIC-to-Switch 的特定场景下，链路级无损仍然有价值，UEC 也保留了 CBFC 作为可选特性。但作为整个网络的流控支柱？PFC 的时代可能真的在走向终结。

**IB 是专用道路上跑的赛车，一切为速度而生。RoCE 是在普通公路上硬改出快车道，靠红绿灯（PFC）和测速摄像头（ECN）来维持秩序。UEC 则是在同一条普通公路上，给车装了自动驾驶（UET 传输层）——车自己知道什么时候该减速、该换道、该停下来。路不需要改，车够聪明就行。**

选哪条路？看你的钱包和你的场景。但有一点是确定的——**无论选哪个，如果你不理解这些流控机制的本质，你的 RDMA 网络一定会在某个深夜把你叫醒。** 而现在，你至少多了一个 UEC 的选项，让那个电话晚几年再来。
