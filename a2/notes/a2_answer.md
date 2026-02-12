## part 1
### 1.a
证明 naive-softmax loss 等于交叉熵损失

交叉熵是用来衡量“两个概率分布”之间的距离：
- $y$ (真实分布)：事实是什么？
- $\hat{y}$ (预测分布)：模型猜是什么？
具体通用公式为： $$H(y, \hat{y}) = - \sum_{i} y_i \log(\hat{y}_i)$$

意思是：把每一个类别 $i$ 的真实概率 $y_i$ 乘以 预测概率的对数 $\log(\hat{y}_i)$，然后加起来取反。

在word2vec中，对于给定的一对词（中心词 $c$，外部词 $o$），我们确信 $o$ 就是那个正确的外部词 
那么意味着“真实分布” $y$ 是一个 One-hot 向量：
- $P(\text{是目标词 } o) = 1$
- $P(\text{是其他任何词 } w) = 0$
假设词表只有 3 个词 [apple, bank, cat]，且真实词是 bank (索引为 1)。那么 $y$ 就是：$$y = [0, 1, 0]$$

代入通用的 Cross-Entropy 公式里。
$$\begin{aligned}
J &= - \sum_{w \in Vocab} y_w \log(\hat{y}_w) \\
&= - \left( y_{apple} \log(\hat{y}_{apple}) + y_{bank} \log(\hat{y}_{bank}) + y_{cat} \log(\hat{y}_{cat}) \right)
\end{aligned}$$
因为 $y_{apple}=0, y_{cat}=0$，这两项直接消失了。因为 $y_{bank}=1$，这一项被保留了。
$$\begin{aligned}
J &= - \left( 0 + 1 \cdot \log(\hat{y}_{bank}) + 0 \right) \\
&= - \log(\hat{y}_{bank})
\end{aligned}$$

结论：当标签是 One-hot 时，交叉熵损失函数 退化 成了 “负的 对数 预测概率”。即：$$J = -\log(\text{预测正确词的概率})$$这正是作业中的公式 (2) ：
$$J_{naive-softmax} = -\log P(O=o|C=c)$$

而具体的$\hat{y}_o$：
在 Word2Vec 里，我们先算一个得分 (Score)：$$s = u_o^T v_c$$

但是得分可以是任意实数（比如 5.0, -2.5），不能直接当概率用。所以我们要用 Softmax 函数把它变成概率：$$\hat{y}_o = \text{Softmax}(s)_o = \frac{\exp(s_o)}{\sum \exp(s_w)}$$所以，把这个代入公式 B：$$J = -\log(\underbrace{\text{Softmax}(u_o^T v_c)}_{\text{就是 } \hat{y}_o \text{，也就是 } P(O=o|C=c)})$$

---

### 1.b
根据作业讲义中的公式，把 $J$ 展开：

$J = -\log \frac{\exp(u_o^T v_c)}{\sum_{w} \exp(u_w^T v_c)} = -u_o^T v_c + \log \sum_{w} \exp(u_w^T v_c)$

分别对两项求导：
- 第一项 $-u_o^T v_c$ 对 $v_c$ 求导很简单，得 $-u_o$。
- 第二项 $\log(\sum \dots)$ 是复合函数，使用链式法则。

令
$S(v_c)=\sum_{w\in V}\exp(u_w^\top v_c)$
则第二项为 $\log S(v_c)$。

对 $v_c$ 求导：
$$\frac{\partial}{\partial v_c}\log S(v_c)=\frac{1}{S(v_c)}\frac{\partial S(v_c)}{\partial v_c}$$

而
$$\frac{\partial S(v_c)}{\partial v_c}=
\sum_{w\in V}\frac{\partial}{\partial v_c}\exp(u_w^\top v_c)$$

对单项 $\exp(u_w^\top v_c)$：
$$\frac{\partial}{\partial v_c}\exp(u_w^\top v_c)
=\exp(u_w^\top v_c)\cdot \frac{\partial}{\partial v_c}(u_w^\top v_c)
=\exp(u_w^\top v_c)\cdot u_w$$

因此
$$\frac{\partial S(v_c)}{\partial v_c}=\sum_{w\in V}\exp(u_w^\top v_c)\,u_w$$

代回去：

$$\frac{\partial}{\partial v_c}\log S(v_c)=
\frac{\sum_{w\in V}\exp(u_w^\top v_c)\,u_w}{\sum_{w\in V}\exp(u_w^\top v_c)}=
\sum_{w\in V}\underbrace{\frac{\exp(u_w^\top v_c)}{\sum_{k\in V}\exp(u_k^\top v_c)}}_{\hat y_w}\,u_w=\sum_{w\in V}\hat y_w\,u_w$$

合并两项导数

$$\frac{\partial J}{\partial v_c}
=
\left(\sum_{w\in V}\hat y_w\,u_w\right) - u_o$$

把 $-u_o$ 写成 one-hot y 的形式：$y_o=1$，其它 0，则
$$u_o=\sum_{w\in V}y_w\,u_w$$
所以
$$\frac{\partial J}{\partial v_c}
=
\sum_{w\in V}(\hat y_w-y_w)\,u_w$$

写成向量化形式:

令 U=[u_1,\dots,u_{|V|}]，则
$$U(\hat y-y)=\sum_{w\in V}(\hat y_w-y_w)\,u_w$$
因此最终：
$$\boxed{\frac{\partial J}{\partial v_c}=U(\hat y-y)}$$

---
### 1.c
- Harmful (有害时):当向量的长度（Magnitude）编码了强度 (Intensity) 或 重要性 (Significance) 信息时。举例：如题目提示 $u_x = \alpha u_y$，如果 $u_{excellent} = 3 \times u_{good}$，归一化后两者变得无法区分。在情感分类中，我们失去了“程度”的信息，导致无法区分强情感和弱情感。
- Not Harmful / Helpful (无害/有益时):当任务只关注语义类别 (Semantic Orientation)，且希望消除词频 (Frequency) 带来的偏差时。解释：如果未归一化，高频词的模长通常很大，会在求和时主导整个句子的向量。归一化后，所有词权重平等，有助于模型捕捉低频但语义丰富的词的信息。
---
### 1.d
对外部词向量 $u_w$ 求导这里要分两种情况讨论：$w$ 是真实词 ($w=o$) 和 $w$ 是干扰词 ($w \neq o$) 。

基础思路：回到 $J = -u_o^T v_c + \log(\sum \exp(u_w^T v_c))$。这次我们要对 $u_w$ 求导。

推导步骤：
- Case 1: 当 $w = o$ (是真实词)
    - 第一项 $-u_o^T v_c$ 包含 $u_o$，求导得 $-v_c$。
    - 第二项 $\log(\dots)$ 里也包含 $u_o$，求导逻辑和 (b) 一样，得到 $\hat{y}_o v_c$。
    - 结果：$(\hat{y}_o - 1) v_c$。
- Case 2: 当 $w \neq o$ (是干扰词)
    - 第一项 $-u_o^T v_c$ 不包含 $u_w$，导数为 0。
    - 第二项 $\log(\dots)$ 里包含 $u_w$，求导得 $\hat{y}_w v_c$。
    - 结果：$\hat{y}_w v_c$。

统一形式：无论哪种情况，结果都可以写成：$$\frac{\partial J}{\partial u_w} = (\hat{y}_w - y_w) v_c$$(因为当 $w \neq o$ 时，$y_w=0$，结果就是 $\hat{y}_w v_c$；当 $w=o$ 时，$y_w=1$，结果就是 $(\hat{y}_o - 1) v_c$)。

---
### 1.e

$$J = -\log \frac{\exp(u_o^T v_c)}{\sum_{w \in Vocab} \exp(u_w^T v_c)}$$

但其实分母里的求和符号 $\sum_{w \in Vocab}$ 遍历了词表里的每一个词。这意味着：$U$ 矩阵里的每一列（每一个 $u_w$），都参与了 $J$ 的计算（主要是在分母里）。

在 Word2Vec 的 Skip-gram 模型中：$u_w$：是某一个具体单词（比如 "apple"）的外部词向量。$U$：是整个词表所有外部词向量组成的大矩阵。

因此，$J$ 是关于整个矩阵 $U$ 的函数。求 $\frac{\partial J}{\partial U}$ 就是问：“如果我们微调矩阵 $U$ 里的任意一个数值，Loss 会怎么变？”

矩阵 $U$ 由列向量组成：$U = [u_1, u_2, \dots, u_{|V|}]$。矩阵的导数 $\frac{\partial J}{\partial U}$，自然也是由每一列的导数组成的：
$$\frac{\partial J}{\partial U} = \left[ \frac{\partial J}{\partial u_1}, \frac{\partial J}{\partial u_2}, \dots, \frac{\partial J}{\partial u_{|V|}} \right]$$

我们在 (d) 题中已经算出，对于任意第 $w$ 列（第 $w$ 个词），它的导数是：
$$\frac{\partial J}{\partial u_w} = (\hat{y}_w - y_w) \cdot v_c$$
(注意：这里 $v_c$ 是一个 $d \times 1$ 的向量，$(\hat{y}_w - y_w)$ 是一个标量数值)。现在我们把它们按列拼起来：
$$\begin{aligned}
\frac{\partial J}{\partial U} &= \Big[ \underbrace{(\hat{y}_1 - y_1) v_c}_{\text{第1列}}, \quad \underbrace{(\hat{y}_2 - y_2) v_c}_{\text{第2列}}, \quad \dots, \quad \underbrace{(\hat{y}_{|V|} - y_{|V|}) v_c}_{\text{最后一列}} \Big]
\end{aligned}$$
会发现每一列都有公共项 $v_c$。

利用矩阵乘法的性质（列向量 $\times$ 行向量 = 矩阵），我们可以把它提取出来，写成外积 (Outer Product) 的形式：
$$\frac{\partial J}{\partial U} = v_c \cdot \underbrace{[\hat{y}_1 - y_1, \quad \hat{y}_2 - y_2, \quad \dots, \quad \hat{y}_{|V|} - y_{|V|}]}_{\text{这是一个行向量}}$$

后面的那个行向量，就是预测分布向量 $\hat{y}$ 减去 真实 One-hot 向量 $y$ 的转置。最终结论：
$$\frac{\partial J}{\partial U} = v_c (\hat{y} - y)^T$$

在做矩阵求导时，最后一步永远是检查维度是否匹配。如果维度对了，答案通常就是对的。

目标：$\frac{\partial J}{\partial U}$ 的形状必须和 $U$ 一样，是 $d \times |V|$（$d$ 是词向量维度，$|V|$ 是词表大小）。

推导的结果：$v_c$ 的形状是 $d \times 1$。$(\hat{y} - y)$ 是列向量 $|V| \times 1$，转置后 $(\hat{y} - y)^T$ 是 $1 \times |V|$。乘法：$(d \times 1) \times (1 \times |V|) = \mathbf{d \times |V|}$。

---
## part2
### 2.a
#### i. Momentum (动量) 的作用
$m$ (梯度的移动平均)，是过去一段时间梯度的“平均值”。

对比：
- SGD (没有动量)：只看当前的梯度。如果当前梯度由一个嘈杂的小批量数据（Minibatch）计算得出，它可能指向错误的方向。这会导致更新路径像“无头苍蝇”一样乱窜（High Variance，高方差）。
- Adam (有动量)：$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$。这意味着当前的更新方向主要由过去积累的惯性决定，当前的梯度 $g_t$ 只是对它做一点微调。

为什么低方差好？
- 它能平滑掉随机噪声。
- 在峡谷（Ravine）地形中（一个方向坡度陡，另一个方向坡度缓），它能抵消在陡峭方向上的震荡，让优化器在这个方向上“冷静”下来，从而在平缓但正确的方向上加速前进。

#### Adaptive Learning Rates (自适应学习率) 的作用

 Adam 更新规则主要包含两步（忽略 $\epsilon$ 和偏差修正）：
 1. 计算梯度平方的滑动平均 ($v$)：
 $$
 v_{t+1} \leftarrow \beta_2 v_t + (1-\beta_2)(\nabla J)^2
 $$
 - 含义：$v$ 记录了该参数在过去一段时间内梯度的大小（Magnitude）。不管梯度是正还是负，平方后都是正的。所以 $v$ 衡量的是这个参数更新得“猛不猛”或者“频繁不频繁”。
 2. 更新参数 ($\theta$)：
 $$\theta_{t+1} \leftarrow \theta_{t} - \frac{\alpha \cdot m_{t+1}}{\sqrt{v_{t+1}}}$$
 - 含义：我们在基础学习率 $\alpha$ 的基础上，除以了 $\sqrt{v}$。这就是自适应（Adaptive）的关键。

那么那种参数会获得较大的更新？
答案：那些梯度较小（Small Gradients）或更新稀疏（Sparse Updates）的参数。

在更新公式中是除以 $\sqrt{v}$。这是一个反比关系。
- 情况 A：常见特征（High Frequency / Large Gradients）
    - 比如 NLP 中的停用词 "the", "and"。它们几乎在每个 batch 里都出现，梯度一直都有值。
    - 累积的 $v$ 会很大。
    - $\text{Update} \propto \frac{1}{\text{大数}}$ $\rightarrow$ 更新步长变小。
- 情况 B：罕见特征（Low Frequency / Small Gradients）
    - 比如生僻词 "idiosyncratic"。它可能几万个样本才出现一次。没出现时梯度为 0，出现时才有梯度。
    - 累积的 $v$ 会很小（因为大部分时间都在累积 0）。
    - $\text{Update} \propto \frac{1}{\text{小数}}$ $\rightarrow$ 更新步长变大。

为什么这对学习有帮助？

直觉 1：平衡更新频率
- 常见特征（"the"）：
    - 因为它出现得太频繁了，如果我们每次都用固定的大学习率更新它，它的参数就会疯了一样地剧烈震荡，很难收敛到精确的最优解。
    - Adam 的策略：既然你经常出现，那我就把你单次的步长调小一点，让你稳步前进，精细微调。
- 罕见特征（"idiosyncratic"）：
    - 它很久才出现一次。如果我们用固定的普通学习率（比如 0.01），那这次更新就像“挠痒痒”一样。等它下次再出现（可能是一万步之后），模型早就把上次学到的那点东西给忘了（被其他参数的更新覆盖了）。
    - Adam 的策略：好不容易抓到你一次！既然你平时不说话，这次说话我就要给你个大喇叭（放大步长），确保模型能深深地记住这次更新的信息。
- 直觉 2：地形适应（归一化曲率）
    - 想象损失函数的形状像一个拉长的峡谷（Taco Shell Shape）。
        -陡峭的方向（对应常见特征）：梯度很大，容易震荡。除以大梯度后，步长变小，防止飞出峡谷。
        - 平缓的方向（对应罕见特征）：梯度很小，容易停滞不前。除以小梯度后，步长变大，加速在平缓区域的移动。
    - Adam 通过除以 $\sqrt{v}$，实际上是在试图把这个狭长的椭圆峡谷，在数学上“拉圆”成一个正圆，让梯度下降在各个方向上的进发速度更加一致。

---
### 2.b
#### 缩放因子 $\gamma$ 的推导

思路：这是一道期望值的计算题。
1. 等式：我们需要 $\mathbb{E}[h_{drop}]_i = h_i$。
2。 代入定义：$\mathbb{E}[\gamma d_i h_i] = h_i$。
3. 提取常量：因为 $\gamma$ 和 $h_i$ 在计算期望时被视为常量（对于掩码分布来说），所以 $\gamma h_i \mathbb{E}[d_i] = h_i$。
4. 消去 $h_i$：$\gamma \mathbb{E}[d_i] = 1$。
5. 计算 $\mathbb{E}[d_i]$：
    - $d_i$ 服从伯努利分布。
    - 取值为 1 的概率是 $1 - p_{drop}$。
    - 取值为 0 的概率是 $p_{drop}$。
    - 期望 $\mathbb{E}[d_i] = 1 \cdot (1 - p_{drop}) + 0 \cdot p_{drop} = 1 - p_{drop}$。
6. 解方程：$\gamma (1 - p_{drop}) = 1 \Rightarrow \gamma = \frac{1}{1 - p_{drop}}$。

##### 为什么要 Train 时用，Evaluation 时不用？
1. Training (为什么要用)：
    - 正则化：防止过拟合，防止神经元共适应（Co-adaptation），即防止某个神经元过度依赖另一个特定的神经元。
    - 数据增强/噪声：引入噪声让模型更鲁棒。

2. Evaluation (为什么不用)：
    - 确定性 (Deterministic)：在测试或生产环境中，我们希望对同一个输入，模型永远给出同一个预测结果。如果开了 Dropout，每次预测结果都不一样，这在工程上是不可接受的。
    - 全能力 (Full Capacity)：测试时我们希望利用整个模型的“完全体”能力，而不是随机的一个“残废”子网络。这对应你笔记里的“模型平均”概念——测试时用全网络近似于所有子网络的平均。

---
## part3
### 3.a 手推一个句子的转移序列
句子：“I presented my findings at the NLP conference”，题面给了前 3 步示例。

初始：
S=[ROOT], B=[I, presented, my, findings, at, the, NLP, conference]
1.	SHIFT → S=[ROOT,I]
2.	SHIFT → S=[ROOT,I,presented]
3.	LEFT-ARC（presented→I）→ S=[ROOT,presented]  （题面已给） ￼

接着（目标依存：findings 是宾语，my 修饰 findings；conference 作为介词短语修饰 presented，at/the/NLP 都依附 conference）：

4.	SHIFT my → S=[ROOT,presented,my]
5.	SHIFT findings → S=[ROOT,presented,my,findings]
6.	LEFT-ARC（findings→my）→ S=[ROOT,presented,findings]
7.	RIGHT-ARC（presented→findings）→ S=[ROOT,presented]
8.	SHIFT at → S=[ROOT,presented,at]
9.	SHIFT the → S=[ROOT,presented,at,the]
10.	SHIFT NLP → S=[ROOT,presented,at,the,NLP]
11.	SHIFT conference → S=[ROOT,presented,at,the,NLP,conference]
12.	LEFT-ARC（conference→NLP）→ S=[ROOT,presented,at,the,conference]
13.	LEFT-ARC（conference→the）→ S=[ROOT,presented,at,conference]
14.	LEFT-ARC（conference→at）→ S=[ROOT,presented,conference]
15.	RIGHT-ARC（presented→conference）→ S=[ROOT,presented]
16.	RIGHT-ARC（ROOT→presented）→ S=[ROOT]，B=[] 完成

总结：

- SHIFT = 搬运工。

- LEFT-ARC = 杀次顶（Stack[-2]）。

- RIGHT-ARC = 杀栈顶（Stack[-1]）。

---
### 3.b
总步数：$2n$ 步。
- 进栈 (Shift)：句子里的 $n$ 个词，每一个都必须从 Buffer 被移入 Stack 一次，这贡献了 $n$ 步 。
- 出栈 (Arc)：每一个词（除了 ROOT）最终都必须作为 Dependent 被连线，并从 Stack 中移除（无论是通过 Left-Arc 还是 Right-Arc），这又贡献了 $n$ 步 
---
