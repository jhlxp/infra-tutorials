
> [MoE](https://zhida.zhihu.com/search?content_id=277377047&content_type=Article&match_order=1&q=MoE&zhida_source=entity) 大模型有几百个专家（Expert），单卡放不下，只能分散到多张 GPU 上。最朴素的做法是均匀平摊——听起来很美。但真实流量里，某几个专家会突然变"热"，承载它的那张卡被打爆，其余卡却闲等。本篇从这个矛盾出发，讲清楚 EEP（[Elastic Expert Parallelism](https://zhida.zhihu.com/search?content_id=277377047&content_type=Article&match_order=1&q=Elastic+Expert+Parallelism&zhida_source=entity)，弹性专家并行）如何在运行时把负载重新摊平：专家迁移怎么搬权重、为什么光搬不够还要复制、[vLLM](https://zhida.zhihu.com/search?content_id=277377047&content_type=Article&match_order=1&q=vLLM&zhida_source=entity)/SGLang 真实系统用什么通信原语、动态扩缩容为什么落在 DP 维度，以及它和 PD 分离扩缩容是什么关系。

### 1\. 引言：均匀分配不等于负载均衡

MoE（Mixture of Experts）模型把一个 FFN 换成 N 个 Expert，每个 token 经过 Router 只激活其中 top-K 个。当 Expert 太多单卡放不下时，用 **专家并行（Expert Parallelism，EP）** 把不同 Expert 放到不同 GPU 上，token 按路由分发过去计算。

EP 的初始放置通常很简单——round-robin 均匀分配。比如 16 个 Expert、8 张卡，每卡 2 个。看起来很均衡。

但这里有个隐藏假设：**每个 Expert 收到的 token 数差不多**。现实并非如此。Router 是学出来的，某些 Expert 在特定输入分布下会被频繁选中（比如代码类 token 总爱去某几个 Expert）。一旦流量里出现这种倾斜：

```text
均匀放置:          热点出现后:
R0: [E0, E8]       R0: [E0=2000, E8=1900]  ← 被打爆
R1: [E1, E9]       R1: [E1=10,   E9=8]     ← 闲等
...                ...
R7: [E7, E15]      R7: [E7=12,   E15=9]    ← 闲等
```

**最忙的卡决定整体延迟，其余卡的算力被浪费。** 静态 EP 对此无能为力——放置在启动时就固定了。

EEP（Elastic Expert Parallelism，弹性专家并行）就是为解决这个问题：**根据实时负载，运行时动态调整 Expert 在 GPU 间的分布**。

本篇回答以下问题：

-   “迁移一个 Expert” 在物理上到底是搬什么？怎么搬？
-   为什么单纯”挪位置”压不平负载，必须”复制热专家”？
-   多个 GPU 进程怎么协同迁移而不死锁？
-   vLLM / SGLang 真实系统用什么通信原语？和教学实现差在哪？
-   为什么动态扩缩容最终是调”数据并行（DP）”而不是”张量并行（TP）”？

* * *

### 2\. 迁移一个 Expert，到底在搬什么

先建立最基本的认知：**一个 Expert 就是几个权重矩阵。**

以 SwiGLU 结构的 FFN 为例，一个 Expert 包含 3 个线性层：

```text
class Expert(nn.Module):
    def __init__(self, hidden, intermediate):
        self.w_gate = nn.Linear(hidden, intermediate, bias=False)  # 门控
        self.w_up   = nn.Linear(hidden, intermediate, bias=False)  # 上投影
        self.w_down = nn.Linear(intermediate, hidden, bias=False)  # 下投影

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

所以”把 Expert E 从 R3（rank 3，即第 3 张卡）迁移到 R0”的物理含义就三步：

1.  **拷权重**：把这 3 个矩阵的张量从 R3 的显存搬到 R0 的显存；
2.  **释放源**：R3 删掉本地这份，释放显存；
3.  **改映射**：全局的 `expert_to_rank` 表里把 E 改成 R0，之后 forward 时 E 的计算就发生在 R0 上。

权重本身不大——hidden=512、intermediate=1024 的小 Expert，3 个矩阵约 157 万参数，fp16 下约 3MB。所以单个 Expert 的迁移在通信上很快，难点不在”搬”本身，而在**让多个 GPU 进程对”搬什么、搬去哪、何时搬”达成一致**，否则集合通信会死锁。

### 2.1 分布式 MoE forward 的基本结构

理解迁移前，先看 EP 下的 forward 是怎么算的。最简单的实现是 **all-reduce 合并**：

```text
def forward(self, x):
    # 所有 rank 收到相同的 x，各自独立算 router（结果一致）
    logits = self.router(x)
    topk_logits, topk_idx = logits.topk(self.top_k, dim=-1)
    weights = F.softmax(topk_logits, dim=-1)

    # 每个 rank 只计算自己持有的 Expert 的贡献，其余位置是 0
    output = torch.zeros_like(x)
    for e, expert in self.local_experts.items():
        mask = (topk_idx == e).any(dim=-1)
        ids = mask.nonzero(as_tuple=True)[0]
        # ... 取出路由到 e 的 token，算出加权贡献累加到 output
        output[ids] += w * expert(x[ids])

    # all-reduce 把 8 张卡的部分结果加起来 → 完整 MoE 输出
    dist.all_reduce(output, op=dist.ReduceOp.SUM)
    return output
```

关键点：

-   所有 rank 拿到**相同的输入 x**，各自独立算 Router（权重一致 → 路由结果一致）；
-   每个 rank **只算自己本地 Expert 的贡献**，没有的位置填 0；
-   最后 `all_reduce(SUM)` 把各卡的部分和加起来，得到完整输出。

> 注意一个容易踩的坑：因为靠 all-reduce 求和合并，必须保证所有 rank 输入的 x 完全一致（用固定 seed 生成），否则各卡算的是不同 batch，加起来就错了。同理，同一个 Expert 在不同 rank 上要用相同的 seed 初始化，保证迁移前后权重一致。

这是 EP 的经典简化模式：不做复杂的 token dispatch/combine（真正的 all-to-all），代价是每张卡都跑了完整 batch 的 Router 和本地 Expert 计算。真实系统（如 DeepEP）会做 token 路由只发给对应 Expert 所在卡，但 all-reduce 版本更适合讲清楚原理。

* * *

### 3\. 第一反应：把热专家”挪走”——以及它为什么不够

热点出现时，最直觉的解法是**迁移（relocation）**：把被打爆那张卡上的热专家搬到空闲的卡。迁移本身就是第 2 节说的那三步——拷权重、释放源、改映射。多进程协同的要点是：**热点检测和迁移决策只在 rank 0 上做**（它持有 all-reduce 后的全局负载），算出”哪个专家从哪搬到哪”后广播给所有 rank，大家拿到完全相同的计划再一起执行，否则集合通信会因步调不一致而死锁。

把这套跑起来：注入热点让 E0、E8 变热（round-robin 下它们都在 R0），然后迁移把这两个热专家分别搬到当前最空的卡。结果却令人沮丧：

| 阶段 | 不均衡度 |
| ----- | ----- |
| 静态基线 | 25.0% |
| 注入热点（全压 R0） | 754.7% |
| 迁移后 | 384.4% |

迁移确实把 R0 腾空了（负载从 971 降到 0），不均衡度从 754.7% 降了一半——**但仍卡在 384.4% 的高位下不去**。

**根因**：一个 Expert 只有**一份权重**，无论搬到哪张卡，那张卡就得独吞它全部的 token。迁移只是把”热”从一张卡平移到另一张卡，**热的总量没变**。更糟的是两个热专家被分到两张不同的卡，于是变成”两张卡各扛一个热专家、其余六张闲等”——当热 Expert 数 ≤ 热点卡数时，挪位置本质上无法降低峰值负载。

这就引出了真正的解法——**复制**。

* * *

### 4\. 冗余复制：把热专家变成多份

### 4.1 逻辑专家 vs 物理专家

要复制 Expert，先要把”专家”这个概念拆成两层——这也是 vLLM/SGLang 的核心抽象：

-   **逻辑专家（logical expert）**：Router 输出的 Expert id，语义固定。比如有 16 个逻辑专家。
-   **物理专家（physical expert）**：真正分布在 GPU 上的权重副本。物理槽位数 = 逻辑专家数 + 冗余数。

```text
num_physical = num_logical + num_redundant
```

关键在于：**一个逻辑专家可以有多个物理副本，分布在不同 rank 上。** 热的逻辑专家副本多，冷的副本少（至少 1 份）。

forward 时，路由到某个逻辑专家的 token，在它的多个物理副本间分摊。最简单的分摊方式是按 token id 做 round-robin：

```text
# 某逻辑专家 lid 的所有物理副本 (owner_rank, slot)
reps = replicas[lid]
# 把这些 token 轮流分到各副本
slot_choice = token_ids % len(reps)
for ri, (owner, p) in enumerate(reps):
    sel = token_ids[slot_choice == ri]   # 分到副本 ri 的 token
    if owner == self.rank:
        # 本卡持有这个副本，计算这部分 token
        ...
```

这样原本压在一张卡上的负载，被切成 N 份摊到 N 张卡——**这是 relocation 做不到、只有 replication 能做到的。**

### 4.2 EPLB 布局算法：[DeepSeek](https://zhida.zhihu.com/search?content_id=277377047&content_type=Article&match_order=1&q=DeepSeek&zhida_source=entity) 风格

如何决定哪个专家复制几份、怎么摆放？这是 EPLB（Expert Parallel Load Balancer）算法的工作。DeepSeek 的开源 EPLB 分两步：

**步骤 1 — 按负载分配副本数（replicate\_experts）**：每个逻辑专家至少 1 份，剩余的冗余名额反复分配给”摊薄后仍最热”的专家：

```text
replica_cnt = [1] * num_logical
for _ in range(num_redundant):
    # 选 负载/当前副本数 最大者，再加一个副本（最小化峰值副本负载）
    lid = max(range(num_logical), key=lambda i: load[i] / replica_cnt[i])
    replica_cnt[lid] += 1
```

**步骤 2 — 贪心打包（balanced\_packing）**：把所有物理副本按”分摊后负载”降序，每个塞进当前最轻的 rank（经典 LPT 贪心）。约束：同一张卡上不放同一逻辑专家两份（同卡复制无法分摊负载）。

### 4.3 效果：负载真正压平了

把热专家复制后再跑，输出直接打印复制原理和实测分摊：

```text
热专家副本数变化: E0: 2→5, E8: 1→5

  冗余复制原理:
    一个 expert 只有一份权重时, 它的 token 必须全压在一张卡上,
    无论怎么搬都搬不平。解法是把热 expert 的权重复制成多份放到
    不同卡 —— forward 时同一逻辑专家的 token 按 token_id % 副本数
    在各副本间 round-robin 分摊。
    E0: 3888 token 原压 1 卡 → 摊到 5 卡, 每卡约 777 token
    E8: 3880 token 原压 1 卡 → 摊到 5 卡, 每卡约 776 token
```

最终效果对比（注：本节是逻辑/物理专家分离的配置，初始放置与第 3 节那套不同，故基线数字不直接可比，但”挪位置 vs 复制”的趋势一致）：

| 阶段 | 不均衡度 |
| ----- | ----- |
| 注入热点 | 382.8% |
| 纯 relocation（只挪位置） | 384.4%（挪走了但压不平） |
| replication（复制热专家） | 71.9% |

复制把不均衡度从 382.8% 压到 71.9%，改善 310 个百分点。

> **残余 71.9% 不是 bug，是固有 tradeoff**。这里 2 个热专家 × 5 副本 = 10 个副本 > 8 张卡，鸽笼原理强制某些卡同时持有两个热副本（E0 和 E8）。当副本总数超过卡数时，无法做到完美均衡——DeepSeek 论文里也存在这个现象。

* * *

### 5\. 真实系统怎么搬：P2P 而非 broadcast

前面只说了迁移要”拷权重”，但具体用哪个通信原语搬，大有讲究。一个朴素的实现是用 `broadcast`（广播）：让源 rank 把权重广播给**所有** rank，只有目标 rank 留下、其余收完即丢。它的好处是简单、不易死锁（集合通信天然同步），但有个明显浪费：**每次迁移所有 rank 都要分配 buffer、参与通信**，哪怕这次迁移和它们毫无关系。8 卡里有 6 张纯属陪跑。

vLLM 和 SGLang 的真实实现都不用 broadcast，而是用**点对点（P2P）的 `batch_isend_irecv`**。结论高度一致：

|  | 朴素 broadcast | vLLM | SGLang |
| ----- | ----- | ----- | ----- |
| 通信原语 | broadcast（全员陪跑） | P2P batch_isend_irecv | P2P batch_isend_irecv |
| 缓冲 | 单 buffer，所有 rank 都分配 | 双缓冲（empty_like） | 双缓冲（empty_like） |
| 无关 rank | 必须参与 | 完全不参与 | 完全不参与 |
| 同卡/不变的专家 | 照样广播 | 跳过网络，本地 copy_ | 跳过网络，本地 copy_ |
| 何时迁移 | stop-the-world | sync 阻塞 / async 后台流 | 生成器分 chunk摊到多次 forward |
| 布局算法 | 简化 EPLB | DeepSeek EPLB | DeepSeek EPLB |

### 5.1 [P2P 迁移](https://zhida.zhihu.com/search?content_id=277377047&content_type=Article&match_order=1&q=P2P+%E8%BF%81%E7%A7%BB&zhida_source=entity)：只有收发双方参与

P2P 的写法是攒一批 `isend`/`irecv` 操作，一起发起再统一等待——vLLM/SGLang 都是这个模式：

```text
p2p_ops = []
for p in range(num_physical):
    if old[p] == new_placement[p]:
        continue  # 该槽专家没变，跳过，不通信
    src_rank, src_slot = pick_source(new_placement[p])  # 找一个持有该专家的源
    if src_rank == dst_rank:
        local_copy(...)   # 同卡：本地 copy，不走网络
        continue
    # 跨卡：dst 准备接收 buffer，src 发送
    if dst_rank == rank:
        buf = [torch.empty_like(t) for t in slot_tensors(p)]   # 双缓冲
        for t in buf:
            p2p_ops.append(dist.P2POp(dist.irecv, t, src_rank))
    if src_rank == rank:
        for t in slot_tensors(src_slot):
            p2p_ops.append(dist.P2POp(dist.isend, t, dst_rank))

# 批量异步发起，再统一 wait —— vLLM/SGLang 同款
reqs = dist.batch_isend_irecv(p2p_ops)
for r in reqs:
    r.wait()
```

实跑时打印每个 rank 实际发起的 P2P 操作数：

```text
迁移规划: 跨卡 P2P 19 个, 本地拷贝 2 个, 不变跳过 3 个
本 rank(0) 实际发起 P2P op: 24 个 (无关 rank 为 0 —— 这就是 P2P 相比 broadcast 的省处)
```

无关 rank 发起 0 个操作——彻底消除了 broadcast”全员陪跑”的浪费。

### 5.2 双缓冲：避免边发边覆盖

两个真实系统都不是原地搬，而是**双缓冲**：跨卡收到的权重先落到 `torch.empty_like` 的临时 buffer，**全部 P2P 完成后**再 copy 回真实权重。

为什么必须双缓冲？因为一个权重 slot 可能”边发边收”——它的旧内容要发给别人，同时要接收新内容。如果原地写，新数据会在发出去之前就把旧数据覆盖掉。朴素的 broadcast + 先发后删侥幸没踩这个坑，但真实的多对多迁移必须双缓冲。

### 5.3 决策一致性：各 rank 各自算，无需广播计划

第 3 节那种”rank 0 决策、再把计划广播出去”的做法能用，但真实系统更进一步：**所有 rank 各自 all-reduce 负载统计**，拿到完全相同的全局负载后，**各自跑同一个确定性算法**算出同一布局——连广播计划都省了。算法都是同一个 DeepSeek EPLB。

### 5.4 不停服务：async / 分 chunk

把推理停下来一次性搬完（stop-the-world）实现最简单，但生产不可接受。真实系统都做到迁移与推理重叠：

-   **vLLM async 模式**：专门起一个后台线程 + 独立 CUDA stream，**每次 forward 只搬一层**，靠跨线程事件握手，主流程几乎不阻塞。
-   **SGLang**：rebalance 是个**生成器**，挂在 `on_forward_pass_end` 上，把迁移**分块 yield** 到多次 forward 里，单 chunk 内 P2P 阻塞，但不正式暂停、不丢请求。

* * *

### 6\. EP 的弹性：当可用 GPU 集合发生变化

前面讲的都是**固定卡数**下重排专家。但真实系统里，”参与 EP 的 GPU 集合”本身会变化，主要有两个方向：

-   **主动扩缩容**：流量涨了加几张卡，流量落了退回去——运维/调度器有意改变卡数；
-   **被动容错**：某张卡突然宕机——非预期地少了卡，系统要自愈。

这两件事**看起来不同，本质同源**：都是”可用 rank 集合变了，专家得在新集合上重新摊开”。vLLM 把前者做深（Elastic EP，主打 DP 维度扩缩容），SGLang 把后者做深（elasticity\_aware，主打故障自愈），但它们共用同一个内核。先讲共同地基，再分两个方向。

### 6.1 共同地基：rank 集合变了 → EPLB 重排 + P2P 迁移

无论是”加了 8 张卡所以 ep\_size 16→24”，还是”挂了 2 张卡所以可用 rank 8→6”，最后都归结为同一件事：

```text
可用 rank 集合变化
   → 在新的 rank 集合上重新算专家放置 (EPLB 算法)
   → P2P 搬权重到位 (第 5 节那套 batch_isend_irecv + 双缓冲)
```

区别只在于喂给 EPLB 的”可用 rank”参数不同：

-   **扩缩容**：`rank_mapping` 把新增/退役的 rank 标出来，专家相应铺开或收回；
-   **容错**：`active_ranks` 把死掉的 rank 标成 -1，专家收回到存活卡。

**这其实是一个连续谱的两端——可用 rank 数变多 or 变少。** 所以本文第 3~5 节讲的迁移机制，在这两个场景里被原样复用，只是入参不同。

### 6.2 主动扩缩容（vLLM）：为什么落在 DP 维度

vLLM 的 Elastic EP 有个值得理解的设计选择：**[弹性维度](https://zhida.zhihu.com/search?content_id=277377047&content_type=Article&match_order=1&q=%E5%BC%B9%E6%80%A7%E7%BB%B4%E5%BA%A6&zhida_source=entity)放在数据并行（DP），而不是张量并行（TP）。** 核心是一个等式：

```text
ep_size = dp_size × tp_size      (tp 固定 ⇒ 调 dp 即调 ep)
```

要理解它，先看 MoE 模型的两类层用**不同的并行方式**：

| 层 | 并行方式 | 含义 |
| ----- | ----- | ----- |
| Attention（含 KV Cache） | DP（数据并行） | 每个 DP rank 持一份完整 attention 权重 + 自己的 KV Cache，处理自己那批请求 |
| MoE / Expert | EP（专家并行） | 全卡的专家合起来才完整，每卡只持 专家数 / ep_size 个 |

所以**一张 GPU 同时扮演两个角色**：它是 1 个 DP rank（管 attention/KV），也是 1 个 EP rank（管一片专家）。EP 组是把 DP 维和 TP 维”拍平”成一个组，所以 EP 大小 = DP × TP。tp 通常固定，于是 **EP 大小完全由 DP 大小决定**。

vLLM 的扩容 API 因此只有一个参数 `new_data_parallel_size`。调一次 DP，**同时发生两件事**：

1.  **加 attention 容量**：多一个 DP 副本 = 多一份 KV Cache、多一条请求处理流 → decode 吞吐上升；
2.  **加 expert 容量**：EP 组同步变大 `tp_size` 个槽位 → 专家（含冗余专家）被 EPLB 重新摊到更多卡 → 单卡专家负载下降（即 6.1 的重排内核）。

**为什么不用 TP 切 attention？** 因为 TP 改切法的代价大得多。根本原因：**TP 是”把一个东西切开分给多卡”，DP 是”整份复制”**——切开的东西改变切法时，所有相关权重和运行时状态都要重新洗牌：

| 维度 | attention 用 DP | attention 用 TP |
| ----- | ----- | ----- |
| attn 权重 | 整份拷贝（加法） | 全员按新 head 边界重切 |
| KV Cache | 新副本从空起步，0 迁移 | 在途 KV 全部按新 head 重切迁移（活数据，无法预热） |
| 单步 forward 通信 | 不变（DP 间几乎不通信） | all-reduce 组大小/通信量全变，带宽敏感 |
| 跨节点 | 可以 | 基本不可行（受 NVLink 域限制） |
| 不停服预热 | 契合 standby 机制 | 难，等于原地给现役卡动手术 |

其中 **KV Cache 这条是质的差别**：TP 切 attention 时 KV Cache 也按 head 切，扩容要把成千上万在途请求的 KV（可达几十上百 GB 的运行时活数据）按新划分重切迁移，且无法提前预热；而 DP 新副本的 KV Cache 从空开始，新请求路由过去自然填充，**存量 KV 零迁移**。

vLLM 用 **standby（预热）机制**做到不停服扩容：旧集群继续服务的同时，按 `new_dp_size` 把更大的进程组提前建好、把权重预传给新卡，最后原子切换，再触发 6.1 的 EPLB 重排把专家摊到新卡。这套之所以可行，正是因为 attention 用 DP（可预热的只读副本）、专家用 EP（静态可预热的权重）——两者都”加法友好”。

> **DP 决定有几个”完整副本单元”，每个单元贡献一个 EP 槽位；调 DP 就同时调了 attention 吞吐和 EP 专家分布。** 因为 `ep_size = dp_size × tp_size` 把两者锁死，DP 成了唯一的弹性旋钮——加卡即加 attention 副本 + 扩 EP 组 + 重摊专家，一步到位。

### 6.3 被动容错（SGLang）：rank 故障的自愈

SGLang 把弹性做到了另一个方向——**容忍 rank 在运行中宕机**。它不是主动改卡数，而是被动检测到某些卡挂了，然后在存活卡上自愈。关键组件：

-   **`elasticity_aware` EPLB 算法**：接收 `active_ranks` 参数，如果存活 rank 数 < 总 rank 数，就只在存活卡上重算一个全局均衡布局，死掉的 rank 位置填零占位。这就是 6.1 内核的”可用 rank 变少”版本。
-   **迁移时自动绕开死 peer**：`_filter_p2p_ops` 在做 P2P 迁移时，丢掉发往/来自死掉 peer 的 `isend`/`irecv` 操作，并记录下”哪些逻辑专家因此丢失”。
-   **从备份恢复丢失的专家**：丢失的专家权重从备份重建。SGLang 用 `expert_backup_manager` 把专家权重注册进 Mooncake 的 CPU 连续缓冲，故障时用 **RDMA `batch_transfer_sync_read`** 把副本拉回来（或从磁盘重载）。
-   **rank 恢复与重入组**：每次 forward 检测死/活 rank，恢复时用 Mooncake `recover_ranks` 重新加入进程组、重置 EPLB 生成器、重广播专家位置 metadata。

注意最后一步——**故障恢复后，SGLang 走的是和正常重排完全相同的流程**（重置 EPLB 生成器、重广播 metadata）。这正印证了 6.1：容错和扩缩容在实现层面是同一套机制的不同入参。

### 6.4 一句话：同源不同向

```text
EP 弹性 (共同地基, §6.1)
                    rank 集合变 → EPLB 重排 + P2P 迁移
                              │
              ┌───────────────┴───────────────┐
              │                               │
        主动扩缩容 (vLLM, §6.2)          被动容错 (SGLang, §6.3)
        "算力够不够"                     "节点还活着吗"
        ep_size 变大/变小                 active_ranks 标记死 rank
        DP 维度 + standby 预热            elasticity_aware + Mooncake 备份
              │                               │
              └──────── 都触发 §6.1 ──────────┘
```

所以这两者**不是并列的二选一选项**，而是同一个能力（弹性 EP）在两个正交方向上的展开：vLLM 解决”主动按需调容量”，SGLang 解决”被动抗节点故障”。它们共享”重算布局 + P2P 迁移”的内核，是一枚硬币的两面。生产系统理想情况下两个方向都要有。

* * *

### 7\. 再上一层：EEP 弹性与 PD 分离扩缩容的关系

读到这里有个自然的疑问：EEP 这套”加卡扩容”，和大家熟悉的 **PD 分离扩缩容** 是一回事吗？会互相打架吗？答案是——它们是**两个正交的旋钮，嵌套使用，互不干扰**。

先一句话交代 PD 分离：大模型推理的两个阶段——Prefill（处理整个 prompt、算出首 token 和 KV Cache）和 Decode（逐 token 生成）——计算特征截然不同（Prefill 算力密集、Decode 访存密集）。把它们拆成**两个独立的模型实例**分开部署、用 RDMA 把 KV Cache 从 P 实例传给 D 实例，就是 PD 分离。每个实例都加载完整模型、各有自己的 EP 组和 EPLB。

关键在于：**P 实例和 D 实例各自都是一个完整的模型实例**，不是”一个实例的两半”。于是扩缩容有了两个层次：

```text
实例级旋钮 (P/D 扩缩容):   要 1 个 P 实例 + 2 个 D 实例?  还是 2P + 3D?
                                  │                       │
                                  ▼                       ▼
实例内旋钮 (EEP):          P 实例内部 DP=2→4         每个 D 实例内部 DP=4→8
                          (P 实例自己的 EP 重摊)     (D 实例自己的 EP 重摊)
```

-   **P/D 扩缩容 = 实例级伸缩**：改变的是”有几个什么角色的实例、各占多大”——P:D 配比怎么调。驱动指标是 P 看 TTFT（首 token 延迟）、D 看 TPOT/吞吐。
-   **EEP 弹性 = 实例内伸缩**：不改实例的数量和角色，只调**某一个实例内部用几张卡**，以及专家在这些卡间怎么分布（即本文第 6 节那套）。

打个比方：**P/D 是”公司要设几个部门、各部门编制多大”，EEP 是”某个部门内部增减人手”**——两个层次的决策，互不替代。

**为什么不干扰**：P 实例和 D 实例是各自独立的进程组 / NCCL 通信域，EP 组不跨实例，EPLB 也是 per-instance 的。对 P 实例做 EEP 扩容只动 P 实例的 EP 组，专家迁移只发生在同一实例的 rank 之间，绝不会”从 P 实例搬专家到 D 实例”。

**而且 PD 分离还让 EEP 更安全**：EEP 重排时虽已用 async / 分 chunk 做轻，仍有轻微抖动。在 PD 分离下，扩 **P 实例**的抖动只影响新请求的 TTFT，正在 decode 的请求（在 D 实例）不受影响；扩 **D 实例**只影响吞吐，prefill 照常。抖动被按阶段隔离了——这是协同，不是冲突。

> **唯一的前提**：上述”正交不干扰”建立在**真正的 PD 分离**（P、D 用各自独立的 GPU 池）之上。若是 P/D co-locate（挤在同一批卡上），两个旋钮就会和”算力给 P 还是给 D”耦合，需要统一调度。

一句话收束：**EEP = 实例内变身材，P/D = 实例间调编制。** 本文从”单卡热点”（微观）讲到”实例内弹性”（中观），PD 分离则是再上一层的”实例间弹性”（宏观）——三层旋钮各管一段。

* * *

### 8\. 总结

```text
1. 问题：均匀分配 ≠ 负载均衡
   - Router 学出来的路由天然倾斜，某些 Expert 会变"热"
   - 静态 EP 在启动时固定放置，热点出现时一张卡被打爆、其余闲等
   - 最忙的卡决定整体延迟

2. 迁移一个 Expert = 搬几个权重张量 + 改一行全局映射
   - 一个 Expert 就是 3 个线性层（SwiGLU），fp16 下单个约几 MB
   - 难点不在"搬"，而在多进程对"搬什么/搬去哪/何时搬"达成一致，避免死锁
   - 决策只在 rank 0 做，再广播计划，保证集体通信步调一致

3. 光挪位置不够，必须复制热专家
   - 一个 Expert 一份权重 → 无论搬到哪，那张卡独吞它全部 token
   - 解法：逻辑专家 / 物理专家分离，热专家复制成多份摊到多卡
   - token 按 round-robin 在副本间分摊
   - 实测：只挪位置仍卡在 384% 高位，复制把不均衡从 382% 压到 72%

4. EPLB 布局算法（DeepSeek 风格）
   - 步骤1 replicate：冗余名额反复给"摊薄后仍最热"的专家
   - 步骤2 balanced_packing：副本按负载降序，塞进当前最轻的卡
   - 残余不均衡是固有 tradeoff：副本总数 > 卡数时无法完美均衡

5. 真实系统（vLLM/SGLang）的关键差异
   - P2P（batch_isend_irecv）而非 broadcast：只有收发双方参与
   - 双缓冲：先收到临时 buffer，全部传完再写回，避免边发边覆盖
   - 跳过不变/同卡的专家：本地 copy 不走网络
   - 不停服：async 后台流 / 生成器分 chunk 摊到多次 forward

6. EP 的弹性：可用 GPU 集合变化的两个方向（同源不同向）
   - 共同地基：rank 集合变了 → EPLB 重排 + P2P 迁移（喂不同的可用 rank）
   - 主动扩缩容（vLLM）：ep_size = dp_size × tp_size，调 DP 一步到位
     · 一张 GPU 双角色：DP rank（attention/KV）+ EP rank（一片专家）
     · 不用 TP 切 attention：TP 要重切权重 + 迁移在途 KV（活数据，无法预热）
       DP 是整份复制、KV 从零填，加法友好，契合 standby 预热
   - 被动容错（SGLang）：elasticity_aware 在存活卡上重排 + Mooncake 备份恢复
   - 二者不是并列选项，是弹性 EP 在"主动调容量 / 被动抗故障"两方向的展开

7. EEP 弹性 vs PD 分离扩缩容：两个正交旋钮，嵌套使用
   - EEP = 实例内变身材（某实例用几张卡 + 专家怎么摊）
   - P/D = 实例间调编制（几个 P 实例 / 几个 D 实例、配比多少）
   - EP 组不跨实例、EPLB per-instance，互不干扰
   - PD 分离反而隔离了 EEP 重排的抖动（扩 P 只影响 TTFT，扩 D 只影响吞吐）
   - 前提：真正的 PD 分离；co-locate 时两旋钮会耦合
```