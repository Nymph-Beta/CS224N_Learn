## Lecture 2（Word Vectors II：GloVe、评估、训练与下游任务）

主要内容：

- Glove：用“全局共现统计”来训练词向量，比起只看局部窗口的word2vec更系统地利用数据
- 如何评估词向量：分为：intrinsic（内在任务）和extrinsic（下游真实任务）
- 下游任务训练： 做分类时可以“只训练分类器”也可以“连词向量一起微调”，但小数据微调会翻车；并引出为什么需要神经网络（讲义第 3 节）。 

---

### 1. Glove究竟解决什么问题

先对比两类旧方法： 

- **Count-based（计数/矩阵分解，如 LSA/HAL）：**

  优点：利用“全局统计”（整个语料的共现）。

  缺点：更偏“相似度”，在 **类比结构**（king - man + woman ≈ queen 这种几何结构）上往往不够好。

- **Window-based（局部窗口预测，如 Skip-gram/CBOW）：**

  优点：能学到复杂语言规律（尤其类比）。

  缺点：主要看局部窗口，本质上“没把全局共现统计吃满”。

**GloVe 的定位**：想要把这两者优点合起来——用全局共现统计，又能学出好用的向量空间结构。

---

### 2. 共现矩阵： Glove的数据是什么样的

- $X_{i,j}$： 词 $j$ 在 $i$ 的上下文窗口出现的次数
- $X_i= \sum_k X_{ik}$： 在 $i$ 的窗口出现过的所有词的总次数
- $P_{ij} = P(w_j\mid w_i) = \dfrac{X_{ij}}{X_i}$：在中心词 $i$ 周围看到 $j$ 的概率

假设语料里，“ice”附近经常出现 “cold”，很少出现 “hot”：

$X_{\text{ice,cold}}$ 大，$X_{\text{ice,hot}}$ 小，那么 $P(\text{cold}\mid \text{ice})$ 就大，$P(\text{hot}\mid \text{ice}$) 就小。

**GloVe 训练就是要把这种统计规律“编码进向量里”。**

---

对于模型，通常使用softmax 参数化模型的条件分布 $Q_{ij}$，再用 NLL/交叉熵让 $Q_{ij}$ 拟合数据的共现分布。。

具体为：$Q_{ij}=\frac{\exp(u_j^\top v_i)}{\sum_{w=1}^W \exp(u_w^\top v_i)}$ ， 其中：

- $\mathbf{v}_i$：中心词 $w_i$ 的向量
- $\mathbf{u}_j$：上下文词 $w_j$ 的向量（另一套 embedding）

> **推导动机（最大熵 / 指数族）**：
>
> 要在词表上定义一个概率分布，且打分函数用“相似度”$s(i,j)=\mathbf{u}_j^\top\mathbf{v}_i$。
>
> softmax 是把任意实数打分 s 变成合法概率分布的标准做法：
>
> $\text{softmax}(s_j)=\frac{e^{s_j}}{\sum_w e^{s_w}}$
>
> 保证：非负、求和为1

训练时：对每个中心词 $i$ 的每个上下文词 $j$，最大化 $\log Q_{ij}$。等价于最小化：

$J=-\sum_{i\in\text{corpus}}\sum_{j\in \text{context}(i)}\log Q_{ij}$

即：*对所有出现过的 (中心词, 上下文词) 对”做负对数似然。*

但同一个 $(i,j)$ 会在语料里重复出现很多次，所以需要合并重复项：

- $\log Q_{ij}$ 这一项出现一次加一次，因为出现了 $X_{ij}$ 次，因此贡献是 $X_{ij}\log Q_{ij}$

于是：$J=-\sum_{i=1}^{W}\sum_{j=1}^{W} X_{ij}\log Q_{ij}$

---

痛点：交叉熵需要 $Q_{ij}$ 是归一化分布，而归一化分母 $ \sum_w \exp(\mathbf{u}_w^\top\mathbf{v}_i)$ 要遍历整个词表，太贵。

（疑问🤔）GloVe思路：先“不归一化”，再用最小二乘

1. 去归一化 

   - 把经验分布 $P_{ij} = X_{ij}/X_{i}$ 的分母去掉，得到 $\hat P_{ij}=X_{ij}$

   - 把模型分布 $Q_{ij}$ 的归一化分母丢掉，得到

     $\hat Q_{ij}=\exp(\mathbf{u}_j^\top\mathbf{v}_i)$

   - 然后用最小二乘拟合两者（并用 $X_i$ 做权重）：

     $\hat J=\sum_{i=1}^W\sum_{j=1}^W X_i\left(\hat P_{ij}-\hat Q_{ij}\right)^2$

​	核心思想：不追求“概率严格的归一化”，而追求“打分能拟合全局共现强度”

2. 从“拟合计数”到“点积拟合 $ \log X_{ij}$”

   $X_{ij}$ 可能非常大，直接最小二乘会被大数值主导，优化困难。于是把误差转到对数域：

   $\hat J=\sum_{i=1}^W\sum_{j=1}^W X_i\left(\log \hat P_{ij}-\log \hat Q_{ij}\right)^2$

   代入 $\hat P_{ij}=X_{ij}$ 与 $\hat Q_{ij}=\exp(\mathbf{u}_j^\top\mathbf{v}_i)$：

   - $\log \hat P_{ij}=\log X_{ij}$
   - $\log \hat Q_{ij}=\log \exp(\mathbf{u}_j^\top\mathbf{v}_i)=\mathbf{u}_j^\top\mathbf{v}_i$

   所以目标变成：

   $\hat J=\sum_{i=1}^W\sum_{j=1}^W X_i\left(\mathbf{u}_j^\top\mathbf{v}_i-\log X_{ij}\right)^2$

   > **GloVe 的核心等式就是让“词向量点积** $\approx$ **共现次数的对数”**。

   这也解释了为什么 GloVe 常被看成一种“加权的矩阵分解”：在分解 $\log X$。

3. 权重 $X_i$ 换成一般的 $f(X_{ij})$

   用 $X_i$ 当权重不一定最优，于是引入更一般的权重函数（可依赖 $X_{ij}$）：

   $\hat J=\sum_{i=1}^W\sum_{j=1}^W f(X_{ij})\left(\mathbf{u}_j^\top\mathbf{v}_i-\log X_{ij}\right)^2$

   - 如果 $X_{ij}=1$：可能是偶然共现，不应该权重大到影响训练

   - 如果 $X_{ij}$ 超大：也不能无限大，否则高频词对目标“碾压式统治”

     所以 $f(\cdot)$ 通常设计成：小计数时权重小，随计数上升而增加，但最终**饱和**。

   > 即：GloVe 在做“带置信度的回归”，共现次数越多越可信，但可信度不会无上限增长。

---

### 3. 词向量如何评估？Intrinsic vs Extrinsic

Intrinsic（内在评估）

- 在一个“简单、快”的代理任务上测embedding
- 目的：快速调参、理解embedding子系统
- 前提：代理任务要和真实下游效果 **正相关**

Extrinsic（外在评估）

- 在真实任务中（情感分类、NER、QA…）
- 缺点：慢；而且如果效果不好，你很难判断是 embedding 的锅还是下游模型/数据/训练策略的原因。

---

Intrinsic 经典：类比（Analogy）到底怎么算？

标准公式为：$ a:b :: c:?$

并用该公式来筛选：$d=\arg\max_i \frac{( \mathbf{x}_b-\mathbf{x}_a+\mathbf{x}_c)^\top \mathbf{x}_i}{\left\|\mathbf{x}_b-\mathbf{x}_a+\mathbf{x}_c\right\|\,\|\mathbf{x}_i\|}$

即：$d=\arg\max_i \cos\big(x_b-x_a+x_c,\ x_i\big)$

> 理想的语义关系是“同一种关系向量相等”：$\mathbf{x}_b-\mathbf{x}_a \approx \mathbf{x}_d-\mathbf{x}_c$
>
> 移项得到：$\mathbf{x}_d \approx \mathbf{x}_b-\mathbf{x}_a+\mathbf{x}_c$
>
> 所以构造目标向量：$\mathbf{t}=\mathbf{x}_b-\mathbf{x}_a+\mathbf{x}_c$
>
> 然后在词表里找最接近 $\mathbf{t}$ 的词向量即可。 

为什么是cosine similarity（而不是点积或欧氏距离）？

cosine similarity：$\cos(\theta)=\frac{\mathbf{t}^\top\mathbf{x}_i}{\|\mathbf{t}\|\,\|\mathbf{x}_i\|}$

它等价于“只比较方向，不比较长度”。词向量的长度常受词频等因素影响；类比更关心“语义方向”。所以用 cosine 更稳健，即“normalized dot-product”

---

类比中哪些超参数最关键？

常调的包括：

- 向量维度（dimension）
- 语料规模（corpus size）
- 窗口大小（window size）
- 窗口是否对称（symmetric vs asymmetric）

从对比表（不同模型/维度/语料）总结了 3 条观察： 

1. **模型选择影响巨大**（GloVe/SG/CBOW/SVD…差异明显）
2. **语料越大通常越好**（见图：数据规模越大准确率越高）
3. **维度太低会欠拟合**（表达不出足够语义因子；讲义用 king/queen/man/woman 需要至少 gender+leadership 作为直觉） 

tips：GloVe 常用窗口大小大约 8（讲义的 Implementation Tip）。

---

还有另一种 intrinsic：和人类相似度打分的相关性

简单的方法是：在一个固定尺度(例如 0‐10)上评估两个词之间的相似度，再和 embedding cosine similarity 做相关性对比。

但是这类指标测试的是“embedding 的相似度排序是否符合人类直觉”，不等价于“下游任务一定更强。

---

多义词怎么处理：一个词多个向量（Multi-prototype）

 Huang et al. (2012) 的思路： 

方法非常直白：

1. 收集某个词所有出现位置的上下文窗口
2. 用上下文词向量的加权平均（比如 idf 权重）表示“这个出现的语境”
3. 对这些语境向量做聚类（spherical k-means）
4. 一个词的不同簇 = 不同“义项”，分别训练不同向量

直觉：把 “bank（河岸/银行）” 这种词拆成 bank#1、bank#2。

---

### 4. 下游任务训练：为外部任务训练

将大多数NLP任务视作分类

用于外部任务的词向量是通过一个在一个更简单的内部任务上优化它们来进行初始化。但是通过使用外部任务进一步训练这些预训练词向量也许能获得更好的性能。

**关键问题：embedding 要不要一起训练？**

 3.2 节结论： 

- **数据集小：不要微调 embedding**（容易把空间结构“扭坏”）

  因为只更新了训练集中出现的少数词，它们在空间中移动后，会导致“没出现的同义词/近义词”相对位置变差，从而测试时掉点。讲义中 “Telly/TV/Television” 的示意图说明：只移动了前两个，第三个没动，但决策边界被带偏，反而错了。 

- **数据集大：可以考虑微调**（覆盖词表更充分，微调更像“任务适配”）

---

**训练通过softmax分类器，输入 $x\in \mathbb{R}^d$，输出属于各类的概率分布**，分类概率为：

$p(y_j=1|x)=\frac{\exp(W_j\cdot x)}{\sum_{c=1}^{C}\exp(W_c\cdot x)}$

其中：

- $x$：输入向量（某个词、或某个样本的表示）
- $W$：分类器权重矩阵，大小大致是 C\times d
- $W_j\cdot x$：第 j 类的“打分”（logit）
- 分母是“归一化”，让所有类别概率加起来=1

如果 $W_j\cdot x$ 比其他类都大很多，$\exp(\cdot)$ 会让它的概率接近 1。

在这里计算词向量 $x$ 属于类别 $j$ 的概率，使用交叉熵函数。即一个训练样本的损失计算为：$\mathcal{L}(\mathbf{x},\mathbf{y})=-\sum_{j=1}^C y_j \log p(y_j=1\mid \mathbf{x})$

---

因为 $\mathbf{y}$ 是 one-hot：只有正确类 $k$ 满足 $y_k=1$，其余 $y_j=0$。

所以求和里只剩一项：

$\mathcal{L}(\mathbf{x},\mathbf{y})=-\log p(y_k=1\mid \mathbf{x})$

再把 $p(\cdot)$ 的 softmax 代进去就得到：$-\log\left(\frac{\exp(\mathbf{W}_k\cdot \mathbf{x})}{\sum_{c=1}^C \exp(\mathbf{W}_c\cdot \mathbf{x})}\right)$

接着数据集上$N$个样本求和：$J=-\sum_{i=1}^N \log\left(\frac{\exp(\mathbf{W}_{k(i)}\cdot \mathbf{x}^{(i)})}{\sum_{c=1}^C \exp(\mathbf{W}_c\cdot \mathbf{x}^{(i)})}\right)$

这里 $k(i)$ 表示第 $i$ 个样本的正确类索引。

---

**softmax + cross-entropy 的梯度**

令 $p=\text{softmax}(z)$，$z=Wx$。

**结论**：$\nabla_{z}\mathcal{L} = p-y$

**推导：

$\mathcal{L}=-\log p_k,\quad p_k=\frac{e^{z_k}}{\sum_c e^{z_c}}$

$\log p_k = z_k - \log\sum_c e^{z_c} \Rightarrow \mathcal{L}= -z_k+\log\sum_c e^{z_c}$

对 $z_j$ 求偏导：

- 若 $j=k：\frac{\partial \mathcal{L}}{\partial z_k}=-1+\frac{e^{z_k}}{\sum_c e^{z_c}}=-1+p_k$
- 若 $j\neq k：\frac{\partial \mathcal{L}}{\partial z_j}=0+\frac{e^{z_j}}{\sum_c e^{z_c}}=p_j$

合并：$\frac{\partial \mathcal{L}}{\partial z_j}=p_j-y_j$

进一步链式法则：

$\nabla_W \mathcal{L} = (p-y)x^\top$

$\nabla_x \mathcal{L} = W^\top(p-y)$

---

**训练时到底更新哪些参数？为什么会过拟合？**

如果在 extrinsic task 里 **既训练分类器权重** $W$，又 **“微调/重训练”词向量** $x$，参数量会爆炸： 

- 训练 $W$：参数量约 $C\cdot d$
- 如果还训练词表里每个词的向量：参数量约 $|V|\cdot d$
- 总参数：$C\cdot d + |V|\cdot d$

因为 $|V|$（词表大小）通常很大（几万到几百万），所以 **特别容易过拟合**（尤其当下游任务训练数据不大时）。

> 这也和 3.2 的结论呼应：数据小就别乱动 embedding，否则会把原来“语义空间”的结构破坏掉。

---

**正则化有什么作用**？

讲义引入 L2 正则（weight decay）：在原损失后面加一项$+\lambda\sum_k \theta_k^2$

这里 $\theta$ 表示“所有可训练参数”（既可能包含 $W$，也可能包含词向量）。 

具体为：$J_{\text{reg}} = -\sum_{i=1}^N \log\left(\frac{\exp(\mathbf{W}_{k(i)}\cdot \mathbf{x}^{(i)})}{\sum_{c=1}^C \exp(\mathbf{W}_c\cdot \mathbf{x}^{(i)})}\right) +\lambda \sum_{m=1}^{C\cdot d+|V|\cdot d} \theta_m^2$

**它的“推导”本质是 MAP（最大后验）视角**：

- 原来的负对数似然对应 $-\log p(\text{data}\mid \theta)$

- 若对参数加零均值高斯先验 $\theta\sim\mathcal{N}(0,\sigma^2 I)$，则

  $-\log p(\theta)\propto \frac{1}{2\sigma^2}\|\theta\|_2^2$

- 合并后就是 $NLL +\lambda \|\theta\|^2$（常数项略去）

讲义用直觉语言说的是：相信参数不该太大（“close to zero”），防止过拟合。

- 没有正则：模型可能把某些权重拉得特别大，以便把训练集拟合到极致
- 加 L2：倾向让参数不要太大（更“平滑”），通常泛化更好

---

**窗口分类：用上下文 disambiguation**

自然语言里**同一个词有多义**，只看词本身经常无法判断含义。讲义用 “sanction” 举例：可以是“允许”也可以是“惩罚”，必须靠上下文区分。 

令窗口半径为 m，中心位置为 i，则输入由拼接构成：

$x^{(i)}_{\text{window}} = \begin{bmatrix} x^{(i-m)}\\ \vdots\\ x^{(i)}\\ \vdots\\ x^{(i+m)} \end{bmatrix} \in \mathbb{R}^{(2m+1)d}$

讲义给了 m=2 的具体拼接例子。 

> 此时线性分类器权重维度也随之变大：$W\in\mathbb{R}^{C\times (2m+1)d}$
>
> 参数量随窗口线性增大 → 更需要正则/更多数据。

反向传播得到对窗口输入的梯度：

$\delta_{\text{window}} = \nabla_{x^{(i)}_{\text{window}}}\mathcal{L} = \begin{bmatrix} \nabla x^{(i-m)}\\ \vdots\\ \nabla x^{(i)}\\ \vdots\\ \nabla x^{(i+m)} \end{bmatrix}$

所以，分类时不输入单个 $x$，而输入一个**窗口**（window）：中心词左右各 $m$ 个词的向量。

- **小窗口**更偏**句法/语法**（syntactic）信息（例如词性、时态）
- **大窗口**更偏**语义**（semantic）信息（主题、场景）

---

**为什么需要非线性分类器/神经网络**

Softmax + 线性打分，本质上是线性决策边界（在特征空间里用一个超平面分开类别）。讲义说：如果数据的最佳分割方式不是线性的，线性分类器就会错很多点。

加入神经网络（至少一层非线性激活），s就能得到**非线性决策边界**，模型容量更强，于是能拟合更复杂的分类关系。讲义用 Figure 9/10 表达：非线性边界能显著减少误分类。 

**最小非线性模型**：

加入一层非线性（MLP）：

$h=\sigma(W_1 x + b_1)$

$p(y\mid x)=\text{softmax}(W_2 h + b_2)$

关键：**非线性来自激活函数** $\sigma$（ReLU/tanh/sigmoid 等）它让模型能表示非线性决策边界（讲义 Figure 10） 

> “为什么只堆线性层没用？”
>
> 因为线性变换的复合仍是线性：$W_2(W_1 x)= (W_2W_1)x$
>
> 没有 $\sigma$ 就不会获得非线性边界。
