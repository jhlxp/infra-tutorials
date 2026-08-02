# 01. DeepGEMM 面试 QA

## Q1：DeepGEMM 这种高性能 GEMM 库，核心应该学什么？

A：

- 不只看 `Linear(x, W) = xW` 这种数学公式；公式简单，系统实现复杂。
- 核心是理解 GPU 上 GEMM 的执行路径：分块、搬运、计算、同步、写回。
- 重点关注 TMA、WGMMA/UMMA、warp specialization、software pipeline、persistent kernel、L2 复用和量化 scale。
- 更进一步，要理解它如何适配 MoE、grouped GEMM、FP8/FP4、MegaMoE 这种业务融合场景。

## Q2：DeepGEMM 为什么快？

A：

- 数据按 tile 组织，避免无序、重复、低效访问。
- TMA 负责大块搬运，减少普通线程搬数据的指令开销。
- WGMMA/UMMA 使用 Tensor Core 执行核心矩阵乘。
- warp specialization 让搬运、计算、epilogue 分工执行。
- software pipeline 让 load、compute、store、scale 尽量重叠。
- persistent kernel 让 SM 长驻并动态领取任务，减少尾部浪费。

## Q3：TMA 的本质作用是什么？

A：

- TMA 是面向 tile 的硬件数据搬运机制。
- 它把大块数据从 HBM/GMEM 搬到片上存储，减少线程手动搬运。
- 它让 producer warp 负责发起搬运，math warp group 专注计算。
- 其价值不是“更快地 load 一个元素”，而是让数据搬运和 Tensor Core 计算可流水化。

## Q4：Warp specialization 的本质是什么？

A：

- 一个 CTA 内的 warp 不再做同一种事，而是按角色分工。
- producer warp 负责 TMA load。
- math warp group 负责 WGMMA/UMMA。
- epilogue 相关 warp 负责 scale、activation、cast、store。
- 本质是把不同硬件单元的工作重叠起来，而不是让所有 warp 在同一个阶段同步等待。

## Q5：Software pipeline 在 GEMM 里解决什么问题？

A：

- GEMM 被切成多个 tile，每个 tile 要经历 load、compute、store。
- 如果串行执行，Tensor Core 会经常等数据。
- software pipeline 用多个 stage 轮转，让当前 tile 计算时，下一个 tile 已经在搬运。
- 本质是用空间换时间，用多级 buffer 隐藏数据搬运延迟。

## Q6：Persistent kernel 为什么适合 MoE？

A：

- MoE expert 的 token 数天然不均匀。
- 静态分配任务容易导致部分 SM 先结束，部分 SM 仍然很忙。
- persistent kernel 让 SM 常驻，持续从任务池领取 tile。
- 本质是把 expert 粒度的不均衡，转化成 tile 粒度的动态调度。

## Q7：FP8 / FP4 为什么需要 scale？

A：

- FP8/FP4 降低带宽和存储压力，也提升 Tensor Core 吞吐。
- 代价是数值范围和精度下降。
- scale 用来恢复量化前的数值范围，通常按 block 或一组元素共享。
- 逻辑上计算的是低精度值乘 scale 后的矩阵乘。

## Q8：Hopper 和 Blackwell 在 DeepGEMM 里的关键差异是什么？

A：

- Hopper 主要使用 WGMMA，scale promotion 更多依赖软件侧流水线隐藏。
- Blackwell 使用 UMMA，并支持 block scaling。
- block scaling 让 scale 更深地进入 Tensor Core 指令路径。
- 本质差异是 scale 处理从软件组织逐渐走向硬件支持。

## Q9：MegaMoE 为什么不是普通 GEMM？

A：

- 普通 GEMM 只关心一次矩阵乘。
- MegaMoE 面向完整 MoE expert MLP 数据流。
- 它涉及 token routing、expert 分组、L1 GEMM、SwiGLU、L2 GEMM、combine。
- 难点不是某一个 GEMM，而是如何把这些阶段融合成高吞吐 pipeline。

## Q10：MegaMoE scheduler 的核心思想是什么？

A：

- 先根据路由结果知道每个 local expert 收到多少 token。
- 再把每个 expert 的 token 按 `BLOCK_M` 切成 token blocks。
- 再按 `BLOCK_N` 切输出维度，形成 GEMM tile。
- 最后让当前 GPU 上的 SM 动态消费这些 tile。
- 本质是把 expert 负载不均衡变成细粒度任务调度问题。

## Q11：MegaMoE 的 wave / phase 应该怎么理解？

A：

- wave / phase 是理解 L1/L2 流水线的调度抽象。
- Phase 1 对应 L1，Phase 2 对应 L2。
- 不应该等所有 L1 全部完成再启动 L2，否则 L2 启动太晚。
- 更合理的是先做 L1 warmup，然后让 L1 和 L2 交错推进。
- 当前实现更接近把 local experts 展开成 pool blocks，再用 task counter 和依赖检查实现流水线。

## Q12：DeepGEMM 和 DeepEP 怎么区分？

A：

- DeepEP 更偏通信数据面，解决 token 在 GPU/rank/expert 之间如何 dispatch 和 combine。
- DeepGEMM 更偏计算数据面，解决 token 到达 expert 后如何高效执行 GEMM。
- MegaMoE 代表两者进一步融合的方向：把路由、expert 计算和 combine 组织成更大的融合 pipeline。

