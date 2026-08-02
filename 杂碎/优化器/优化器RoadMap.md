# 深度学习优化器入门：它不仅在选方向，也在定义“距离”和“时间”

> 把 SGD、AdamW、Muon 排成一条不断升级的版本链很诱人，但这也是理解优化器最容易犯的第一个错误。
>
> SGD 改变的是**梯度如何被估计**；Momentum 改变的是**如何使用历史信息**；Adam 改变的是**不同坐标如何被缩放**；AdamW 改变的是**正则化如何进入更新**；Muon 改变的是**矩阵参数采用什么几何**；Hyperball 则把**权重半径和更新尺度**从隐式动力学变成显式约束。
>
> 它们不是在回答同一个问题，也不是简单的前后替代关系。更准确的理解方式，是把优化器看成一组可以组合的设计模块。

---

## 0. 一个统一视角：优化器由什么组成

设训练目标为

$$
F(\theta)
=\frac{1}{N}\sum_{i=1}^{N}\ell_i(\theta)+R(\theta),
$$

其中 $R(\theta)$ 可以是显式正则项，也可以为空。优化器直接看到的是训练目标及其梯度，而不是测试集泛化误差。所谓“某个优化器更容易泛化”，说的是它的更新规则对最终解产生了某种**隐式偏置**，而不是它直接优化了泛化能力。

在第 $t$ 步，可以把训练过程拆成五个问题：

1. **梯度估计**：使用全量数据，还是 mini-batch？
2. **时间记忆**：只看当前梯度，还是累积历史信息？
3. **参数几何**：什么样的参数变化算“一样大”？按坐标、按层，还是按矩阵奇异方向衡量？
4. **更新时钟**：学习率、batch size、动量和参数范数共同决定模型在函数空间里走得多快。
5. **约束与正则**：是否衰减参数、固定范数、投影到某个集合，或者裁剪更新？

抽象地写，一个训练步可以表示成

$$
\begin{aligned}
g_t &= \text{GradientEstimator}(\theta_t),\\
s_t &= \text{Memory}(s_{t-1},g_t),\\
u_t &= \text{Geometry}(g_t,s_t,\theta_t),\\
\theta_{t+1} &= \text{Constraint}\bigl(\theta_t,-\eta_tu_t\bigr).
\end{aligned}
$$

这套分解比“GD → SGD → AdamW → Muon”的单线叙事更准确。AdamW、MuonW、Muon Hyperball 等方法，本质上都是不同模块的组合。

一个更凝练的说法是：

> **优化器既在选择方向，也在定义距离；学习率、batch size 和权重范数则共同定义训练的时间尺度。**

---

## 1. 从 GD 到 SGD：差别不只是计算量

### 1.1 Full-batch Gradient Descent

全量梯度下降在每一步使用全部样本：

$$
\nabla F(\theta_t)
=\frac{1}{N}\sum_{i=1}^{N}\nabla \ell_i(\theta_t)
+\nabla R(\theta_t),
$$

并更新

$$
\theta_{t+1}=\theta_t-\eta_t\nabla F(\theta_t).
$$

它的优势是当前数据集上的梯度估计稳定，缺点是每一步代价高，而且在现代大规模训练中并不一定能获得更好的时间效率或泛化表现。

这里还要注意：即使遍历了全部训练样本，只要训练中存在随机数据增强、Dropout 或其他随机机制，实际计算仍可能带有随机性。

### 1.2 Mini-batch SGD

设 $B_t$ 是第 $t$ 步采样的 batch，随机梯度为

$$
g_t
=\frac{1}{|B_t|}\sum_{i\in B_t}\nabla \ell_i(\theta_t).
$$

在理想的独立均匀采样下，条件期望满足

$$
\mathbb E[g_t\mid \theta_t]=\nabla L(\theta_t),
$$

于是可以写成

$$
g_t=\nabla L(\theta_t)+\xi_t.
$$

但不要把 $\xi_t$ 想成简单的、各向同性的白噪声。真实训练中的梯度噪声通常具有以下性质：

- 它依赖当前位置 $\theta_t$，因此是**状态相关**的；
- 不同方向上的方差不同，因此是**各向异性**的；
- shuffle without replacement 会让相邻 batch 的噪声相关；
- 数据增强、序列打包、负采样和分布式聚合都会改变它的统计结构。

在独立有放回采样的简化模型中，若单样本梯度协方差为 $\Sigma(\theta_t)$，则

$$
\operatorname{Cov}(g_t\mid\theta_t)
=\frac{1}{|B_t|}\Sigma(\theta_t).
$$

若从有限数据集无放回采样，还会出现 finite-population correction。因而“方差与 batch size 成反比”是一个有条件的近似规律，而不是无条件定理。

### 1.3 梯度累积到底等价于什么

假设使用 $K$ 个等大的 micro-batch，参数在累积期间保持不变，并且最终使用平均梯度：

$$
g_{\mathrm{acc}}
=\frac{1}{K}\sum_{k=1}^{K}g_{B_k}(\theta_t).
$$

如果损失对样本可加、缩放方式一致，那么

$$
g_{\mathrm{acc}}
=g_{B_1\cup\cdots\cup B_K}(\theta_t),
$$

也就是说，它等价于**一次更大有效 batch 的 SGD 更新**，而不等价于连续做 $K$ 次小 batch 更新。

因此更准确的表述是：

- 梯度累积仍然可以属于 mini-batch SGD；
- 它可能等价于一次大 batch 更新；
- 它不等价于参数在每个 micro-batch 后都变化的 $K$ 次 SGD。

这种等价还依赖若干条件。以下情况会让累积梯度与真正的大 batch 出现差异：

- BatchNorm 使用了不同的 micro-batch 统计量；
- 每个 micro-batch 分别做梯度裁剪；
- loss 没有正确除以 accumulation steps；
- 混合精度溢出处理在中途丢弃了部分梯度；
- 学习率调度、weight decay 或 EMA 按 micro-step 而不是 optimizer step 更新；
- 非加性损失需要跨样本或跨 batch 计算统计量。

所以工程上真正应该核对的，不是“有没有使用 gradient accumulation”这一句话，而是：**一次 optimizer step 到底聚合了哪些样本、进行了哪些状态更新。**

---

## 2. SGD 的噪声为什么有时有益，但不能简单归结为“寻找平坦极小值”

### 2.1 一个有用但不完整的直觉

小 batch 带来的随机性会改变优化轨迹。它可能帮助参数离开某些狭窄区域，降低对训练样本细节的过度适配，并形成不同于确定性梯度下降的隐式偏置。

早期研究观察到，小 batch 训练有时更容易到达在参数扰动下较稳定的区域，而大 batch 训练更容易出现泛化差距。这催生了“平坦极小值泛化更好”的经典解释。

这个解释可以保留，但必须加上限定：

> **SGD 噪声可能偏好某些更稳定、曲率更温和的解；但普通参数空间中的 sharpness 不是一个与参数化无关的泛化指标。**

同一个网络函数可以通过参数重缩放被表示成任意“更尖锐”或“更平坦”的形式，而预测行为完全不变。因此，不能把“SGD 一定找到平坦极小值，平坦极小值一定泛化更好”当作定理。

更稳妥的理解是：SGD 的噪声结构会和局部曲率、样本梯度协方差、模型对称性以及学习率共同作用。它影响的是**优化路径与解的选择偏置**；“平坦性”只是描述这种现象的一种坐标相关视角。

### 2.2 大 batch 不等于必然泛化差

大 batch 会降低梯度估计噪声，但最终效果还取决于：

- 学习率是否随 batch size 合理缩放；
- 是否使用 warmup；
- 总更新次数和训练 token 数是否足够；
- 正则化、数据增强和调度器是否重新调优；
- batch 是否已经超过当前任务的 critical batch size。

大量实验表明，大 batch 的泛化差距可以在相当程度上通过更合适的训练配方缩小，甚至消除。因此，batch size 更像是在权衡两种效率：

- **样本效率 / token efficiency**：达到目标损失需要多少数据；
- **时间效率 / parallel efficiency**：达到目标损失需要多少并行时间。

超过 critical batch size 后，继续增大 batch 往往能提高硬件并行度，却不能同比减少优化步数，于是 token efficiency 开始下降。

### 2.3 batch size 和学习率必须一起看

只说“batch 越小，噪声越大”是不够的。参数真正受到的随机扰动还乘上了学习率。在简化的随机微分方程视角中，常见的噪声尺度近似包含

$$
\text{noise scale}
\propto
\eta\left(\frac{N}{B}-1\right)
\approx \frac{\eta N}{B},
\qquad B\ll N.
$$

Momentum 还会进一步改变有效时间尺度。因此，以下操作可能产生相似但不完全相同的效果：

- 降低学习率；
- 增大 batch size；
- 改变 momentum；
- 改变每个样本被访问的次数。

“训练后期逐渐增大 batch”是一种可行策略，它在某些设置下可以部分替代 learning-rate decay，并提高并行效率；但它不是普遍适用的训练规律。

---

## 3. Momentum：不只是给梯度降噪

一种常见写法是

$$
m_t=\beta m_{t-1}+(1-\beta)g_t,
$$

$$
\theta_{t+1}=\theta_t-\eta_t m_t.
$$

也有实现使用

$$
v_t=\beta v_{t-1}+g_t,
$$

再用 $\theta_{t+1}=\theta_t-\alpha_t v_t$。两种写法只差一个可吸收到学习率中的尺度，但这意味着：**比较不同代码库的 momentum learning rate 时，必须先检查具体 convention。**

Momentum 的作用至少有三层：

1. **时间平滑**：偶发、相互抵消的梯度分量会被削弱；
2. **持续加速**：多步一致的方向会累积速度；
3. **改善病态曲率下的轨迹**：在狭长谷底中，它可以减少高曲率方向上的来回震荡，同时沿低曲率方向推进。

因此，把 Momentum 只描述为“历史梯度降噪”会漏掉它最重要的优化意义：它在改变整个系统的动力学。

$\beta$ 还决定了记忆长度。粗略地说，指数滑动平均的时间窗口约为

$$
\frac{1}{1-\beta}.
$$

例如 $\beta=0.9$ 对应约十步的记忆尺度，$\beta=0.99$ 则接近百步。Nesterov momentum 又进一步使用“向前看”的梯度或等价组合，常常能改善响应速度。

---

## 4. 从 AdaGrad、RMSProp 到 Adam：自适应方法到底在适应什么

把 Momentum 直接跳到 Adam，会丢掉一条关键思想线：**按坐标缩放梯度。**

### 4.1 AdaGrad：累计历史平方梯度

AdaGrad 为每个坐标累计平方梯度：

$$
v_t=v_{t-1}+g_t^2,
$$

$$
\theta_{t+1}
=\theta_t-\eta\frac{g_t}{\sqrt{v_t}+\varepsilon}.
$$

它对稀疏特征很有效：频繁出现的坐标会逐渐减小步长，少见坐标保留较大步长。缺点是 $v_t$ 单调增长，长期训练中有效学习率可能衰减过度。

### 4.2 RMSProp：只记住近期尺度

RMSProp 用指数滑动平均代替无限累积：

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

它估计的是近期的逐坐标梯度平方尺度。

### 4.3 Adam：一阶动量加逐坐标二阶矩

Adam 再加入一阶矩：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

经过偏置修正后，

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},
$$

更新为

$$
\theta_{t+1}
=\theta_t-
\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}.
$$

这里最容易出现三个误解。

#### 误解一：$v_t$ 是梯度方差

不是。$v_t$ 是**未中心化二阶矩**的指数估计，也就是近似 $\mathbb E[g^2]$，而不是

$$
\operatorname{Var}(g)=\mathbb E[g^2]-\mathbb E[g]^2.
$$

因此，某个坐标的 $v_t$ 大，既可能因为噪声大，也可能因为它长期具有较大的确定性梯度。

#### 误解二：Adam 是二阶优化器

Adam 使用“二阶矩”，但它没有计算 Hessian，也没有建模跨坐标曲率。更准确地说，它是一个**一阶、对角预条件**方法。

#### 误解三：Adam 不再需要全局学习率调度

逐坐标自适应不等于全局训练时钟消失。Adam 仍然高度依赖 base learning rate、warmup、decay schedule、$\beta_1$、$\beta_2$ 和 $\varepsilon$。

Adam 的优势是对梯度坐标尺度较不敏感，并能处理稀疏、非平稳或尺度差异大的梯度。它的局限是只使用对角统计，无法描述参数之间的相关结构；原始 Adam 在某些构造出的凸问题上也可能不收敛，AMSGrad 等变体正是针对这类理论问题提出的。

---

## 5. Weight Decay：先把 L2 正则和参数衰减分开

### 5.1 L2 regularization 是修改目标函数

若在目标中加入

$$
F_{\mathrm{reg}}(\theta)
=F(\theta)+\frac{\lambda}{2}\|\theta\|_2^2,
$$

则梯度变为

$$
\nabla F_{\mathrm{reg}}(\theta)
=\nabla F(\theta)+\lambda\theta.
$$

对于没有 momentum、使用标量学习率的普通 SGD，更新为

$$
\theta_{t+1}
=(1-\eta_t\lambda)\theta_t
-\eta_t\nabla F(\theta_t).
$$

在这个受限情形下，L2 regularization 与 multiplicative weight decay 可以通过系数换算得到相同轨迹。

一旦加入 momentum 或自适应预条件，这种简单等价通常就不再成立，因为 $\lambda\theta$ 会进入优化器状态。

### 5.2 为什么 Adam 中的 coupled L2 不等于 weight decay

如果把 $\lambda\theta_t$ 直接加到 Adam 的梯度里，那么它会同时进入

- 一阶矩 $m_t$；
- 二阶矩 $v_t$；
- 逐坐标自适应分母。

于是“让参数向零收缩”的作用，也被历史梯度尺度重新缩放。不同坐标受到的实际正则强度不再相同。

### 5.3 AdamW 的正确更新

AdamW 只用 loss gradient 更新矩估计。设

$$
u_t=\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon},
$$

则现代常见记号下，AdamW 为

$$
\boxed{
\theta_{t+1}
=(1-\eta_t\lambda)\theta_t-
\eta_tu_t
}
$$

也可以写成

$$
\theta_{t+1}
=\theta_t-
\eta_tu_t-
\eta_t\lambda\theta_t.
$$

不要把它写成

$$
(1-\eta_t\lambda)
\bigl(\theta_t-\eta_tu_t\bigr),
$$

因为后者会多出一个 $\eta_t^2\lambda u_t$ 交叉项，并不严格等价。

AdamW 的核心不是“Adam 加了一个正则项”，而是把两件事分开：

- $u_t$：根据训练损失选择更新方向；
- $(1-\eta_t\lambda)$：独立控制参数收缩。

还要注意，每一步的衰减强度实际由 $\eta_t\lambda$ 决定。学习率调度变化时，累计 weight decay 也随之变化。

### 5.4 不是所有参数都应该相同地 decay

Bias、LayerNorm/RMSNorm gain、某些 embedding 与输出参数的范数可能具有直接语义。实践中经常对它们设置 `weight_decay=0`，但这是一种需要验证的参数分组策略，而不是绝对规则。

---

## 6. Scale invariance：必须先说清楚“哪些参数”真的尺度无关

### 6.1 精确定义

对某个参数块 $W$，若在其余参数固定时满足

$$
L(cW)=L(W),\qquad c>0,
$$

才称这个参数块对正尺度变换不敏感。

一个典型近似例子是

$$
y=\operatorname{Norm}(Wx),
$$

在忽略 normalization epsilon 等细节时，$W\mapsto cW$ 可能不改变输出。

但不能因为网络使用了 LayerNorm 或 RMSNorm，就断言所有矩阵都 scale-invariant。例如

$$
y=W\operatorname{Norm}(x)
$$

通常会随着 $W$ 的缩放而变化；残差连接也会让分支幅值与 skip connection 的相对比例具有实际意义。现代 Transformer 中更常见的是局部、近似或联合参数重缩放对称性，而不是每个权重矩阵都严格满足 $L(cW)=L(W)$。

### 6.2 精确尺度不变会带来两个结论

对 $L(cW)=L(W)$ 关于 $c$ 在 $c=1$ 处求导，可得

$$
\langle W,\nabla_W L(W)\rangle_F=0.
$$

即梯度与权重的径向方向正交。

同时，梯度满足缩放关系

$$
\nabla L(cW)=\frac{1}{c}\nabla L(W).
$$

这两个式子比笼统地说“归一化让权重范数不重要”更有信息量：函数可能不依赖半径，但优化速度依赖半径。

### 6.3 为什么 SGD 会让范数增长

若当前随机梯度也严格满足 $\langle W_t,g_t\rangle_F=0$，则

$$
\begin{aligned}
\|W_{t+1}\|_F^2
&=\|W_t-\eta_tg_t\|_F^2\\
&=\|W_t\|_F^2+
\eta_t^2\|g_t\|_F^2.
\end{aligned}
$$

因此，纯切向的有限步更新仍会因为勾股关系增加参数半径。

但这个结论不能无条件推广到 Momentum 和 Adam：历史动量是在过去位置计算的，未必与当前 $W_t$ 正交；自适应缩放也可能改变切向性。

### 6.4 角度变化与“有效学习率”

令

$$
\widehat W=\frac{W}{\|W\|_F}.
$$

小步长下，方向的一阶变化为

$$
\Delta \widehat W
\approx
-\frac{\eta_t}{\|W_t\|_F}
\left(
g_t-
\langle \widehat W_t,g_t\rangle_F\widehat W_t
\right).
$$

若 $g_t$ 已经完全切向，则角度移动大小近似为

$$
\|\Delta\widehat W\|_F
\approx
\frac{\eta_t\|g_t\|_F}{\|W_t\|_F}.
$$

这个量更准确地叫**angular update**。如果比较两个表示同一函数的参数 $W$ 与 $cW$，又因为梯度会缩小为 $1/c$，那么 SGD 在归一化坐标中的有效步长通常呈现

$$
\eta_{\mathrm{eff}}
\propto
\frac{\eta_t}{\|W_t\|_F^2}
$$

的缩放关系。

因此，“角度移动”与“把参数重缩放后对应的有效学习率”不是完全相同的概念。不同优化器对参数缩放的响应也不同；Adam 的逐坐标归一化会改变上述幂次关系。

### 6.5 重新理解 weight decay

对于严格**函数级** scale-invariant 的参数块，沿该缩放对称性改变 $\|W\|$ 不会改变表示函数；若只在当前 loss 上近似不变，这个结论也只能作为局部动力学近似。因此，这类参数上的 weight decay 主要作用可能不是传统意义上的容量正则，而是：

- 抵消随机切向更新带来的范数增长；
- 控制参数旋转速度；
- 让不同层或神经元进入较稳定的 rotational equilibrium；
- 间接决定 angular learning rate。

对于不具备尺度不变性的参数，weight decay 仍然会直接改变模型函数，也仍可能发挥传统正则化作用。两种机制不能混为一谈。

---

## 7. Hyperball：显式控制半径和更新尺度，而不只是“投影回球面”

Hyperball 是一个可以包裹 Adam、Muon 等 base optimizer 的方法。它最关键的地方有两个：

1. 固定权重矩阵的 Frobenius norm；
2. 固定 base optimizer update 的 Frobenius norm。

设 base optimizer 给出的更新为 $u_t$，定义

$$
\operatorname{Normalize}_F(X)
=\frac{X}{\|X\|_F},
$$

并令固定半径

$$
R=\|W_0\|_F.
$$

Hyperball 的更新是

$$
\boxed{
W_{t+1}
=R\cdot
\operatorname{Normalize}_F
\left(
W_t-
\eta_tR\cdot
\operatorname{Normalize}_F(u_t)
\right)
}
$$

因此它不是简单地计算 $W_t-\eta_tu_t$ 后再投影。它还先把 $u_t$ 归一化，使未投影 trial step 的范数固定为 $\eta_tR$。

这带来一个很清晰的解释：

- $R$ 决定权重半径；
- $\eta_t$ 直接控制相对更新尺度；
- base optimizer 主要负责提供方向和内部状态；
- 投影负责移除径向漂移。

令 $\widehat u_t=u_t/\|u_t\|_F$，在小步长下可展开为

$$
\widehat W_{t+1}-\widehat W_t
\approx
-\eta_t
\left(
\widehat u_t-
\langle \widehat W_t,\widehat u_t\rangle_F\widehat W_t
\right).
$$

这说明球面归一化在一阶近似下相当于把更新投影到切空间。换句话说：

> **Hyperball 的投影不只是“防止范数变大”，它还在过滤 base update 的径向分量。**

这也解释了为什么 trial step 长度固定，并不意味着实际球面位移或测地角度完全固定；实际方向变化仍取决于 $u_t$ 与 $W_t$ 的夹角。

Hyperball 的设计动机是把 weight decay 隐式完成的“半径—角速度校准”改成显式控制。近期实验显示，它在部分 Qwen3-style 预训练设置中改善了学习率跨宽度、深度的迁移，并增强了 Muon 在更大训练规模下的收益。

不过它仍应被视为较新的研究方向，而不是已经普遍取代 weight decay 的默认方案。当前证据集中在特定 Transformer 架构和训练配方；它也只适合施加在范数冗余或近似冗余的矩阵上。Embedding、normalization gain 和其他范数具有语义的参数，通常仍交给普通优化器处理。

---

## 8. 从 element-wise sign 到 matrix sign：真正的统一线索是“选择什么范数”

### 8.1 Steepest descent 先要定义一步有多大

给定梯度 $G$，考虑线性化后的局部问题：

$$
\min_{\Delta}
\langle G,\Delta\rangle
\quad
\text{s.t.}
\quad
\|\Delta\|_{\mathcal G}\le \rho.
$$

这里的范数 $\|\cdot\|_{\mathcal G}$ 定义了什么叫“同样大小的一步”。换一个范数，最陡下降方向就会改变。

- 使用向量 $\ell_2$ 或矩阵 Frobenius norm，方向与 $-G$ 相同；
- 使用 element-wise $\ell_\infty$ norm，最优方向是 $-\operatorname{sign}(G)$；
- 对矩阵使用 operator/spectral norm，最优方向是矩阵的 polar factor，也就是优化器文献中常说的 matrix sign。

因此，signSGD 和 matrix sign 的联系，不是“标量 sign 升级成矩阵 sign”这么简单，而是：

> **它们分别是不同参数几何下的最陡下降方向。**

没有哪一种几何对所有参数都天然更高级。关键是它是否匹配参数在网络中的功能。

### 8.2 element-wise sign 保留了什么

signSGD 使用

$$
\Delta=-\eta\operatorname{sign}(g).
$$

它忽略每个坐标的幅值，只保留正负号。它的原始优势之一是通信压缩；从几何上看，它限制每个坐标的最大变化量相同。

它并不建模坐标间相关性，因此对矩阵权重来说，行空间、列空间和奇异方向都没有被显式利用。

### 8.3 matrix sign / polar factor

设矩阵 $G\in\mathbb R^{m\times n}$ 的 compact SVD 为

$$
G=U_r\Sigma_rV_r^\top,
$$

其中 $r=\operatorname{rank}(G)$。定义

$$
\operatorname{msign}(G)
=U_rV_r^\top.
$$

它把所有非零奇异值变成 1：

$$
U_r\Sigma_rV_r^\top
\quad\longrightarrow\quad
U_rI_rV_r^\top.
$$

严格地说，它保留的是左右奇异子空间，而不是原矩阵在 Frobenius 向量空间中的“方向”。当奇异值差异很大时，matrix sign 会显著改变原更新方向。

例如

$$
G=10u_1v_1^\top+0.1u_2v_2^\top
$$

会被变换为近似

$$
\operatorname{msign}(G)
=u_1v_1^\top+u_2v_2^\top.
$$

这有两面性：

- 它避免大奇异值方向支配全部更新；
- 它也可能放大原本很弱、甚至主要由噪声构成的奇异方向。

因此，“压平奇异值”本身不是无条件收益。Momentum、正则化、近似精度和更新尺度都决定了它最终是改善条件数，还是放大噪声。

对于矩形或秩亏矩阵，“正交化”更准确地说是得到 semi-orthogonal / partial-isometry 更新。秩亏情况下还需要约定零奇异值和零空间上的处理。

---

## 9. Muon：Momentum 加谱几何，而不是一个神秘的二阶优化器

Muon 的名字来自 **MomentUm Orthogonalized by Newton–Schulz**。它主要用于神经网络隐藏层的二维矩阵参数。

一个理想化版本可以写成：

$$
M_t
=\beta M_{t-1}+(1-\beta)G_t,
$$

$$
O_t
\approx
\operatorname{msign}(M_t),
$$

$$
W_{t+1}
=(1-\eta_t\lambda)W_t
-\eta_t s_{m,n}O_t,
$$

其中 $s_{m,n}$ 是与矩阵形状有关的更新缩放。实际实现还常使用 Nesterov 风格的 momentum combination。

### 9.1 Newton–Schulz 在做什么

直接做 SVD 得到 $UV^\top$ 太贵。Muon 使用若干轮只包含矩阵乘法的 Newton–Schulz 多项式迭代，近似计算矩阵 zeroth power / polar factor，因此更适合 GPU。

实践中的常用系数并不追求高精度地把所有奇异值收敛到 1，而是优先让少量迭代在 bfloat16 下快速得到“足够平坦”的奇异值谱。也就是说，工程版 Muon 的更新通常只是**近似 matrix sign**，而不是精确 SVD 结果。

### 9.2 为什么先做 Momentum，再做 matrix sign

如果直接对单个 noisy gradient 做 matrix sign，微小奇异方向可能被强行放大。先用 Momentum 聚合多步信息，可以提高弱方向中信号相对于噪声的稳定性，然后再压平奇异值。

这提供了一种合理解释，但不是一般性保证。Muon 的收益仍然依赖任务、batch、参数尺度和实现。

### 9.3 为什么必须有 shape scaling

若 $O=UV^\top$ 的秩为 $r$，则

$$
\|O\|_F=\sqrt r.
$$

因此不同形状的矩阵即使使用同一个全局学习率，Frobenius update norm 也会不同。Muon 实现通常加入与行列比有关的缩放；后续大规模工作还进一步校准每类参数的更新尺度。

所以，省略 shape scaling 会让“Muon 更新公式”在概念上不完整。

### 9.4 Muon 应该用在哪些参数上

Muon 通常用于 hidden 2D weights。以下参数往往仍由 AdamW 或其他普通优化器处理：

- token embedding；
- 最终输出 head；
- bias；
- normalization gain；
- 一维或标量参数。

这不是实现细节，而是方法定义的一部分：不同张量承担不同功能，未必应该共享同一种几何。

### 9.5 Muon 与 Shampoo 的关系

Shampoo 为矩阵或高阶张量维护结构化的左右预条件器。若去掉历史累积，并在满秩或伪逆意义下对当前矩阵梯度做理想化的左右 inverse-fourth-root 预条件，可得到

$$
(GG^\top)^{-1/4}
G
(G^\top G)^{-1/4}
=UV^\top.
$$

这说明 matrix sign 与结构化预条件有深刻联系。但完整 Shampoo 会累积二阶统计，不能简单说 Muon 就是“廉价 Shampoo”。

更准确的定位是：

> **Muon 是一种一阶、矩阵结构感知、采用 spectral-norm 几何的优化方法；它不计算 Hessian，也不等于传统二阶法。**

---

## 10. 优化器不是一条进化链，而是一张设计地图

更好的 RoadMap 不是

$$
\text{SGD}
\rightarrow
\text{AdamW}
\rightarrow
\text{Muon}
\rightarrow
\text{Hyperball},
$$

而是下面这张表：

| 设计问题 | 代表方法 | 它真正改变的东西 |
|---|---|---|
| 如何估计梯度 | GD、mini-batch SGD、gradient accumulation | 噪声、单步成本、并行度 |
| 如何使用历史 | Momentum、Nesterov | 时间平滑与动力学 |
| 如何按坐标缩放 | AdaGrad、RMSProp、Adam | 对角预条件与坐标尺度 |
| 如何处理大 batch | LR scaling、warmup、LARS、LAMB | 层级更新比例和并行效率 |
| 如何加入参数收缩 | L2、SGDW、AdamW | 正则项是否进入优化器状态 |
| 如何利用张量结构 | Shampoo、SOAP、Muon | 跨坐标或矩阵奇异方向几何 |
| 如何控制半径与角速度 | weight decay、norm projection、Hyperball | 径向动力学和相对更新尺度 |
| 如何控制数值稳定性 | gradient clipping、loss scaling、epsilon | 溢出、异常梯度和有限精度行为 |

这些模块可以组合：

- **AdamW** = 一阶 EMA + 逐坐标二阶矩 + decoupled weight decay；
- **MuonW** = Momentum + matrix sign approximation + shape scaling + decoupled weight decay；
- **Muon Hyperball** = Muon 提供 base direction，Hyperball 固定 update norm 和 weight norm；
- **SOAP** = 在结构化预条件器的特征基中运行 Adam 式更新。

一旦使用这张地图，很多争论会变得更清楚。比如“Muon 是否优于 AdamW”并不是一个纯粹的方向比较：如果两者的 relative update、weight norm、weight decay、学习率 schedule 和参数分组没有对齐，实验同时改变了**几何和训练时钟**。

---

## 11. 一个更合理的学习顺序

### 第一阶段：先掌握 SGD 的基本动力学

需要回答：

- 梯度为什么是下降方向？
- learning rate 太大或太小会怎样？
- mini-batch gradient 在什么条件下无偏？
- gradient accumulation 与连续小步更新有什么差别？

### 第二阶段：用二次函数理解 Momentum

重点不是背公式，而是观察：

- 高曲率方向为什么震荡；
- 低曲率方向为什么推进缓慢；
- Momentum 如何改变特征方向上的收敛速度。

### 第三阶段：学习 AdaGrad、RMSProp、Adam

需要明确：

- $v_t$ 是未中心化二阶矩，不是 Hessian，也不是方差；
- 逐坐标缩放解决了什么，又丢失了什么相关结构；
- Adam 为什么仍需要全局 schedule。

### 第四阶段：区分 L2 与 weight decay

需要能够独立写出：

- coupled L2 的更新；
- decoupled weight decay 的更新；
- AdamW 的正确公式；
- 为什么 momentum 和 adaptive preconditioning 会破坏简单等价。

### 第五阶段：理解 normalization 下的径向—切向分解

需要掌握：

- 什么条件下 $L(cW)=L(W)$；
- 为什么梯度与 $W$ 正交；
- angular update 与 effective learning rate 的区别；
- weight decay 为什么可能主要在控制旋转速度。

### 第六阶段：从“坐标”走向“矩阵范数”

建议依次理解：

- element-wise $\ell_\infty$ geometry 与 sign update；
- Frobenius norm 与普通梯度方向；
- operator norm 与 matrix sign；
- Shampoo 的结构化预条件；
- Muon 为什么在 Momentum 后做近似 polar decomposition。

### 第七阶段：再看 Hyperball 等显式约束方法

此时应该把它理解成一个 optimizer wrapper：它不负责创造新的梯度信息，而是重新规定 base update 的尺度和参数轨迹所在的流形。

---

## 12. 实践中比“换优化器名字”更重要的检查项

在比较优化器前，至少应对齐或记录以下量：

1. **optimizer step 数和训练 token 数**：相同 epoch 不代表相同更新次数；
2. **learning-rate schedule 与 warmup**：base LR 相同也不代表累计移动相同；
3. **有效 batch size**：包括数据并行和 gradient accumulation；
4. **参数分组**：哪些参数使用 weight decay，哪些参数交给 Muon；
5. **relative update**：$\|\Delta W\|/\|W\|$；
6. **angular update**：$\|\widehat W_{t+1}-\widehat W_t\|$；
7. **gradient clipping rate**：大量触发裁剪时，实际优化器已被 clipping 规则主导；
8. **optimizer-state memory 与通信开销**：矩阵优化器的额外计算未必是主要瓶颈，分片和通信可能更关键；
9. **时间到目标质量**：同时看 step efficiency、token efficiency 和 wall-clock efficiency；
10. **超参数迁移能力**：小模型上调好的学习率能否迁移到更宽、更深的模型。

一个实用原则是：

> **先把 AdamW 或 SGDM 基线调正确，再一次只替换一个设计模块。**

否则，优化器名称只是实验中许多变化的标签，难以判断收益究竟来自方向、尺度、范数控制，还是训练预算。

---

## 13. 最后总结

1. SGD 的随机性不是简单白噪声，而是状态相关、各向异性的梯度估计误差。
2. Gradient accumulation 可以等价于一次大 batch SGD，但不等价于多次连续小 batch 更新。
3. “SGD 偏好平坦极小值”是有启发性的经验解释，不是与参数化无关的普遍定理。
4. Batch size 不能脱离 learning rate、momentum 和训练预算单独讨论。
5. Momentum 不只是降噪，它会改变病态曲率下的整体动力学。
6. Adam 的 $v_t$ 是逐坐标未中心化二阶矩；Adam 仍是一阶对角预条件方法。
7. AdamW 的正确形式是 $(1-\eta\lambda)\theta-\eta u$，而不是把完整 Adam 更新后再整体相乘。
8. Scale invariance 必须针对具体参数块判断；LayerNorm/RMSNorm 不会自动让所有矩阵尺度无关。
9. 对 scale-invariant 参数，weight decay 可能主要控制半径、角速度和 rotational equilibrium。
10. Hyperball 同时固定 weight norm 和 base update norm；只写“更新后投影回球面”会漏掉其核心。
11. Matrix sign 保留奇异子空间并压平非零奇异值；它是 operator-norm 几何下的最陡方向，而不是普通 sign 的简单升级版。
12. Muon 是 Momentum、近似 matrix sign、shape scaling 与参数分组的组合；它是矩阵几何方法，不是 Hessian 二阶法。
13. 优化器的发展更像一张多轴设计地图，而不是一条从旧到新的淘汰链。

---

## 延伸阅读与论文线索

### SGD、batch size 与泛化

- Keskar et al., *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima*, ICLR 2017, arXiv:1609.04836。
- Dinh et al., *Sharp Minima Can Generalize for Deep Nets*, ICML 2017, arXiv:1703.04933。
- Smith & Le, *A Bayesian Perspective on Generalization and Stochastic Gradient Descent*, ICLR 2018, arXiv:1710.06451。
- Smith et al., *Don’t Decay the Learning Rate, Increase the Batch Size*, ICLR 2018, arXiv:1711.00489。
- Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*, arXiv:1706.02677。
- McCandlish et al., *An Empirical Model of Large-Batch Training*, arXiv:1812.06162。

### Momentum、Adam 与 weight decay

- Sutskever et al., *On the Importance of Initialization and Momentum in Deep Learning*, ICML 2013。
- Duchi et al., *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*, JMLR 2011。
- Kingma & Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015, arXiv:1412.6980。
- Reddi et al., *On the Convergence of Adam and Beyond*, ICLR 2018 / arXiv:1904.09237。
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019, arXiv:1711.05101。

### Scale invariance 与角度动力学

- van Laarhoven, *L2 Regularization versus Batch and Weight Normalization*, arXiv:1706.05350。
- Roburin et al., *Spherical Perspective on Learning with Normalization Layers*, arXiv:2006.13382。
- Li, Lyu & Arora, *Reconciling Modern Deep Learning with Traditional Optimization Analyses: The Intrinsic Learning Rate*, NeurIPS 2020, arXiv:2010.02916。
- Kosson, Messmer & Jaggi, *Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks*, arXiv:2305.17212。

### Sign、矩阵几何与 Muon

- Bernstein et al., *signSGD: Compressed Optimisation for Non-Convex Problems*, ICML 2018, arXiv:1802.04434。
- Gupta, Koren & Singer, *Shampoo: Preconditioned Stochastic Tensor Optimization*, ICML 2018, arXiv:1802.09568。
- Bernstein & Newhouse, *Old Optimizer, New Norm: An Anthology*, arXiv:2409.20325。
- Vyas et al., *SOAP: Improving and Stabilizing Shampoo using Adam*, arXiv:2409.11321。
- Jordan et al., *Muon: An Optimizer for Hidden Layers in Neural Networks*, 2024。
- Liu et al., *Muon is Scalable for LLM Training*, arXiv:2502.16982。

### Hyperball

- Wen et al., *Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization*, arXiv:2606.16899。

