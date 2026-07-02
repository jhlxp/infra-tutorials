
在研究强化学习训练框架 veRL 时，初学者首先会被灌输 single controller 和 multi-controller 这两个基础的概念。此概念最早可以追溯到 Google 在 2022 年发布的一篇旧文 Pathways，veRL 在设计中也深受 Pathways 文章的影响。

今日重读此文，一些死去的记忆开始攻击我。

2021年11月，Google 发布了Jeff Dean 写的博客《[Introducing Pathways: A next-generation AI architectur](https://link.zhihu.com/?target=https%3A//blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/)e》。此文是Pathways 的首次亮相，它被成为next-generation AI architecture。2022 年初，Google 发布了540B 大模型 [PaLM](https://link.zhihu.com/?target=https%3A//www.jmlr.org/papers/volume24/22-1144/22-1144.pdf)，正是使用 Pathways 训练而成。随后 Pathways 的论文预印本同步亮相，揭露了更多细节，后面发表在 MLSys 22 中。值得一提的是Pathways 的尾作Yonghui Wu，正是如今字节Seed 的负责人。

2022年，当时鲜有人相信模型增大能产生神奇效果。Meta 的 在 3 月份发布开源模型 175B OPT，可惜效果很差，一定程度影响了大众对大模型的印象。相比 DeepMind Gopher 和 OpenAI 的 GPT3.5，Google 540B 在规模上是遥遥领先的，因而 Google 在当时AI 系统的认知处于绝对前沿地位。对时代背景感兴趣的同学，可以翻阅笔者口述历史《[方佳瑞：大模型Infra这些年，从黑铁时代到黄金时代再到白银时代](https://zhuanlan.zhihu.com/p/708594043)》。

但是，对于彼时阅读 Pathways 论文的我来说，读完的感受确实“如读”。我觉得这是做流水线并行问题的工作，是专门为JAX + TPU设计的，在现实场景中似乎没有太多实际意义。 这主要是我当时认识不足导致的。

当时国内（甚至是全世界）对 Pathways 论文理解最为透彻的是 oneflow 团队。他们发表了两篇 Pathway 解读文章，第一篇《[OneFlow：解读谷歌Pathways架构（一）：Single-controller与Multi-controller](https://zhuanlan.zhihu.com/p/495592456)》，第二篇更加激进，直接喊出《[Pathway 是向前一步是 OneFlow](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/N99dRgFYC9zOOcGlg0Ulsw)》。oneflow 是当时国内罕见的同时在思考下一代 AI 系统的团队，下面摘录一段他们文章原文：

> 我可能是比较能理解 Pathways 的少数人之一。Pathways 讨论的这些问题我们都思考过，好几年前就思考了，而且还研发出 OneFlow，也写过论文探讨这些问题。遗憾的是，我们在论文里讨论这些问题时，几乎从来没被理解过。今天，从 Google 论文讲出这些道理，这就是不需要证明的真理了。

这就像裴松之注三国志，金圣叹批水浒传，B 站 up 主解读大明王朝1566，oneflow 团队的点评比 Pathways 原文还要精彩。这里有高山流水的惺惺相惜，也有曲高和寡的意难平。

时光荏苒，三年半的时间匆匆过去，和当下的 AI 系统对比，非常值得回味 Pathways 在当年的一些预言，和 oneflow 的注解。让我们透过 Pathways 一窥在那个大模型的蛮荒时代，从业者对未来 AI 系统的绮梦。

### Single vs Multi-controller

作为一套 Google 内部涵盖软件和硬件的庞大系统，要说清楚 Pathways 是是非常复杂的，这里牵扯很多 TPU 特殊的设计。这里只能尽可能讲清楚Pathways 所希望传递的普世理念，也是后世 veRL 等系统所汲取的源泉。

在 2022 年，以 PyTorch 为代表的 ML 训练系统是 MPI 式的，每个卡启动一个进程，所有进程运行的代码完全一样，然后插入 AllReduce/ReduceScatter/AllGather等集合通信原语，来实现分布式的 DNN 训练。按照Flynn分类法，这就是典型的 SPMD并行范式，Pathways 称之为 multi-controller。每个进程都是分布式训练的一个控制器，平等的执行相同程序。

**multi-controller** 对于执行数据并行（DP）程序简直再合适不过，这也是 2022 年最主流的并行方式。但是 Google 认为，下一代模型会变得复杂，会采用非 SPMD 也就是 MPMD 的方式运行，那 multi-controller 将无法胜任。程序将是以 MPMD 方式计算图，图每个节点是是一个多卡运行的 SPMD 程序，每张卡的执行程序是不同的。

Pathways 认为什么样的程序是 MPMD 呢？ 作者提到了 Pipeline Parallel（PP），MoE 两个场景。PP 中每个 Stage 使用不同参数，计算逻辑可能不同。MOE 中每个 token 路由到每个专家是不确定的，因此每次计算图都不同，程序是非 MPMD 的。按照后世的经验，二者用 SPMD 也并非不行。对于 PP，Megatron-LM 1F1B 复杂流水线并行还不也用 torchrun 跑的好好的。对于 MoE 模型，采用 EP/TP 等并行，MoE 训练和Dense 似乎也并没有什么范式级别的变化。

所以 PP 和 MoE 并没有击中 multi-controller 的死穴。反而是后来强化学习，多个模型构成复杂的计算图才是multi-controller 不能承受的生命之轻，单这是 Pathways 作者们始料未及的。 RLHF 论文是 2022 年 3 月发表的，真正被意识到其威力，要等到年末的 ChatGPT 发布了。

为了解决 multi-controller 对 MPMD 的限制，Pathways 使用 **single-controller**，系统通过一个 controller（master 节点）来描述计算图，计算图的每个节点是是一个多卡运行的 SPMD 程序，每个节点点采用 multi-controller 方式运行。single-controller 编程灵活性，能够实现用单一的 Python 进程管理成千上万个 TPU 设备。灵活编程只是一方面，更重要的是，single-controller 可以让每个计算图节点的资源可以动态调节。multi-controller 的每个进程所拥有的计算资源在计算图执行过程是一成不变的，而 single-controller 可以让图的每个节点资源都不同。

single-controller 是沿用分布式计算（Distributed Computing）的思路，用 master-slave 方式组织分布式的计算，计算资源可以是异构的、动态变化的。而 multi-controller 则是沿用高性能计算（HPC）的思路，在超计算机上跑并行任务，计算资源是同构的。

TensorFlow V1 早期采用 single-controller 的思路设计，它会把计算图切分，然后插入 send recv 节点来实现流水线式的分布式训练。不过面对简单的 DP，这种过度设计逐渐式微，并很快被 PyTorch 打败。Pathways 作者认为未来的任务都是类似 PP、MoE 高度稀疏和动态的（如今的 RL 任务），此时的过度设计反而变得恰到好处。

### Dive into single-controller

如何实现单控制器（Single-controller）方式的计算图，Pathways 采用了谷歌内部的闭源数据流执行引擎 PLAQUE。至于开源方案，很自然会考虑使用 Ray。计算图中的每个节点对应一个 Ray Actor，Actor 使用 PyTorch 以单程序多数据（SPMD）方式进行计算。 这样扩展到大规模，在实现层面还是有很大挑战。

首先是 Single-controller 的调度难度。我认为 Pathways 消除调度开销优化主要为是 TPU 定制的。和 cuda kernel 可以自动支持 dynamic shape 不同，TPU 根据输入 shape JIT 地编译成可执行程序再发给 TPU 上运行，这导致很大的 kernel launch的开销。另外，cuda kernel 可以很细碎，通过互相抢占；TPU kernel 是一个大程序，从头运行到底，在分布式运行时很容易造成死锁。所以 Pathways 提出了 Parallel Async Dispatch 和 Gang Scheduling Dispatch 来解决这两个问题。

那么 GPU 上是否就没有这些调度问题呢？笔者认为 CUDA kernel launch 足够小的调度开销，如果不使用 torch.compile这种编译优化，不搞 auto parallel 这些“华而不实”的自动搜索，调度开销是可以忽略。但是，死锁问题还是可能存在，这一点 oneflow 的第一篇文章对 NCCL 死锁问题有很精彩的论述。

Single-controller 系统上计算图节点之间数据传输都经过中心化的 controller 节点。这里很很容易成为瓶颈，从而影响系统的扩展性。Pathways 提到了使用和 Ray 类似的 sharded object store 支持 HBM 等设备内存管理。object store 会以用户透明的方式，自动管理数据对象。但是在大规模并行上有很大问题，oneflow 团队有一个精彩论述：

> 另外，把 Ray 用到深度学习系统里还是有一个不易察觉的大“陷阱”，那就是 Ray 通过 object store 和 RPC 机制实现了一套“自动”的数据搬运机制 （object 在 object store 之间的隐式迁移），本意当然是帮上层用户着想，让他们不用再操心复杂的数据搬运，但这个“自作主张”，“越俎代庖”之举在深度学习里会帮倒忙，不是做少了，而是做过了。  
> 多说一句，做 AI 芯片的朋友对这一点体会最深，通用 CPU 都使用 cache 机制，硬件自动完成预取的工作，让软件开发更简单，但仍存在 cache miss rate 的问题，AI 芯片不能承受这个代价，再加上深度学习任务有显著的规律，所以更希望让软件显式的管理缓存何时预取数据，这里普遍使用一种叫 scratchpad 的技术，以实现100%的”命中“。

这一点在 verl 中也会遇到，在大规模并行时，两个控制分多 GPU 的 Ray Actor 之间传输很大的文件，比如视频、图片，object store的透明调度差强人意。Ray 又缺少 RDMA 传输机制，这会导致潜在的性能问题。

最后把 single controller 所控制的节点抽象成 sharded dataflow现在已经是见怪不怪了，你用 3D 并行启动一个训练任务就是 sharded dataflow，不过当时还是颇为新奇的观点。

### Single Controller, Multiple Controller and More

veRL的流行让 single-controller 设计重新回到人们的视线。那是否不适合 MPI-style 实现的分布式程序，都要用 single-controller 设计呢？

显然，分布式系统世界并非只有single-controller 和 multiple-controller的非此即彼。因此，并不能死记硬背下 SPMD 用 multi-controller，MPMD 用 single-controller。

有些 MPMD 程序还是需要用微服务的方式来组织。尤其是编程灵活性不首要任务，需要很在意调度开销的程序。比如， PD 分离的 LLM 推理引擎，VAE\\TextEncoder\\DiT 服务组成 Diffusion Model推理引擎，RL 训练系统 Nemo-Aligner 都是采用类似微服务方式设计。这种方式下，计算图的每个节点并需要指导全局的计算图，只需要进行和前置取数据+计算+后置节点翻数据的事件循环，就可以完成自己的任务。

随着AI 系统变得越来越复杂，我认为大家有三种系统设计上的选择。对于 SPMD，优先选择 multi-controller。对于 MPMD，如果每个节点计算和通信比例小，很在意调度开销，用微服务架构。如果调度开销比较小，用 single-controller。

最后发个招聘贴：

火山引擎机器学习平台（veMLP）招聘 veRL 开源开发实习生 ，负责字节 veRL 开源建设与维护。对 veRL 开发感兴趣的，想加入正规军的，且有时间全职实习的。

简历发送至邮箱：[wangjerry@bytedance.com](mailto:wangjerry@bytedance.com)
