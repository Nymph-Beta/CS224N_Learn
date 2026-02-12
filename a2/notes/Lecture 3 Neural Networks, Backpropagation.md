## Lecture 3 Neural Networks, Backpropagation

主要内容：

- **为什么需要神经网络**： 

  线性分类器（比如直接 softmax/逻辑回归）只能画线性边界；真实 NLP 特征组合常常是非线性的，所以需要“**线性变换 + 非线性激活**”堆起来，形成弯曲的决策边界。 

- **神经网络在算什么（前向）**：

  输入向量 $x$（在 NLP 里经常是“窗口内词向量拼接”） → 线性变换得到  $z$ → 激活得到隐藏表示 $a$ → 再线性组合得到打分 $s$。

- **怎么定义“做得好/坏”（损失）**：讲义用的是**最大间隔损失**：希望真样本分数 $s$ 比假样本 $s_c$ 至少大 1。 

- **怎么训练（反向传播）**：用链式法则把 $\frac{\partial J}{\partial \theta}$ 高效算出来。关键产物是每层的误差信号 $\delta$，以及两个最重要的公式：

  - 矩阵梯度：$\nabla W^{(k)} = \delta^{(k+1)} {a^{(k)}}^\top$
  - 误差回传：$\delta^{(k)} = f'(z^{(k)}) \circ (W^{(k)\top}\delta^{(k+1)}) $

- **训练技巧**：梯度检查、正则化、dropout、激活函数选择、数据预处理、Xavier 初始化、学习率/动量/自适应优化等。

---

### 1. 神经元与单层：从 logistic 回归到“并行的多个 detector”

神经网络时具有非线性决策边界的分类器。现在要看是如何实现的这一点。

#### 1.1 单个 sigmoid 神经元

接收 $n$ 个输入产生一个输出。输入是一个向量 $x\in\mathbb{R}^n$，参数是权重 $w\in\mathbb{R}^n$ 和偏置 $b\in\mathbb{R}$，先做线性组合，再过非线性（sigmoid）得到输出：

$a=\sigma(w^\top x+b)=\frac{1}{1+\exp(-(w^\top x+b))}$

可以理解为：

1. **线性部分**：$t=w^\top x+b$（把 n 个输入“加权求和再平移”成一个标量）
2. **非线性部分**：$a=\sigma(t)$（把标量压到 (0,1) 之间，让模型能表达非线性边界）

> 讲义还写了把 bias 合并进向量的写法：把 $[x;1]$ 与 $[w;b]$ 做点积（等价于把 bias 当作“输入恒为 1 的权重”）。 
>
> 即偏置 $b$ 被视作连接到固定输入“1”的权重。
>
> 可以把 bias 看作是第 $n+1$ 个 weight，从而把所有运算统一成一个整洁的矩阵乘法（Matrix Multiplication），而不需要单独写一步加法。

#### 1.2  A Single Layer of Neurons（单层：把多个神经元堆起来）

现在把“一个 sigmoid 单元”扩展成“**m 个** sigmoid 单元并行”：

- 同一个输入 $x\in\mathbb{R}^n$ 喂给 m 个神经元
- 第 $i$ 个神经元有自己的 $w^{(i)}\in\mathbb{R}^n、b_i\in\mathbb{R}$，输出 $a_i$

讲义把它写成矩阵形式：

- $W\in\mathbb{R}^{m\times n}$：把每个 $w^{(i)\top}$ 作为一行堆起来

- $b\in\mathbb{R}^m$

- 先算

  $z=Wx+b \quad (\Rightarrow z\in\mathbb{R}^m)$

  再逐元素过 sigmoid 得到

  $a=\sigma(z)=\sigma(Wx+b)\quad (\Rightarrow a\in\mathbb{R}^m)$

> 这一层在做:
>
> “ $m$ 个不同的特征探测器”。每个神经元的$ w^{(i)}$ 学到一种“输入特征组合”，输出$a_i$ 就像“这种组合是否出现/强弱”。

------

#### **1.3 Feed-forward Computation (前馈计算：从特征提取到决策)**

##### **1. 为什么要引入隐层？(Motivation: Non-linear Interactions)**

- **线性模型的局限**：线性分类器（如 Softmax Regression）通常独立地处理输入的每个维度。
- **NLP 中的挑战 (NER 案例)**：
  - 例子："Museums in Paris are amazing"。任务是判断中心词 "Paris" 是否是命名实体 。
  - 单纯看到 "Museums" 或 "Paris" 可能不足以判断。模型需要捕捉 **“词与词之间的交互” (interactions between the words)**。
  - **非线性决策**：例如，“只有当 'Museums' 是第一个词 **且** 'in' 是第二个词时，才强烈指向后面是地名” 。这种条件逻辑（AND/OR/XOR）是线性模型无法直接表达的。
- **结论**：我们需要一个中间层（Hidden Layer）来把原始输入“揉”在一起，提取出能表达这些交互关系的高级特征。

##### **2. 数学架构与维度 (Architecture & Dimensions)**

讲义构建了一个最简单的两层神经网络（单隐层 + 输出层）：

- **前向计算三步走** ：
  1. **线性映射**：$z = Wx + b$ （把输入投射到隐层空间）
  2. **非线性激活**：$a = \sigma(z)$ （产生特征/激活值）
  3. **线性打分**：$s = U^T a$ （综合特征得出结论）
- **维度示例** ：
  - 输入 $x \in \mathbb{R}^{20}$ （例如窗口大小 5 $\times$ 词向量维度 4）。
  - 隐层单元数 $m=8$ （即 8 个特征探测器）。
  - **权重矩阵** $W \in \mathbb{R}^{8 \times 20}$，偏置 $b \in \mathbb{R}^8$。
  - **激活向量** $a \in \mathbb{R}^8$。
  - **输出权重** $U \in \mathbb{R}^{8 \times 1}$，最终得分 $s \in \mathbb{R}$。

##### **3. 核心解读：权重与特征的物理意义 (Deep Interpretation)**

这是理解神经网络“黑盒”的关键。讲义提到：“*One can think of these activations as indicators of the presence of some weighted combination of features.*” 

- **权重 $W$ 是什么？——“模板”或“过滤器”**
  - $W$ 的每一行代表一个特定的探测模式。
  - **数值含义**：
    - $W_{ij}$ 很**大且为正**：神经元非常“喜欢/关注”输入的第 $j$ 个特征。
    - $W_{ij}$ 很**大且为负**：神经元非常“讨厌/抑制”输入的第 $j$ 个特征。
    - $W_{ij} \approx 0$：神经元忽略该特征。
  - **来源**：权重不是人设定的，而是**学出来的**（Learned）。初始化时是随机噪声，通过反向传播（Backpropagation）和梯度下降，根据 Loss 自动调整出来的。
- **激活值 $a$ 是什么？——“特征存在的指示器” (Feature Presence Indicator)**
  - $z = Wx + b$ 计算的是输入与模板的匹配程度（得分）。
  - Sigmoid 函数 $\sigma(z)$ 充当**门控/开关**：
    - 若 $a \approx 1$：**灯亮了**。表示探测器发现了它寻找的模式（Feature is **Present**）。
    - 若 $a \approx 0$：**灯灭了**。表示输入中没有该模式（Feature is **Absent**）。
  - **本质**：$a$ 是网络自动提取出的“高级特征”（如“介词短语结构”、“大写单词模式”等）。

##### **4. 第二层矩阵 $U$ 的作用：分工逻辑**

为什么得到 $a$ 之后还要乘一个 $U$？

- **角色分工——“调查员”与“法官”**：
  - **Layer 1 ($W, b$) 是调查员**：它的任务是去案发现场（输入 $x$）寻找各种线索，并汇报哪些线索存在（输出 $a$）。
  - **Layer 2 ($U$) 是法官**：它的任务不是看原始现场，而是根据调查员的报告（$a$）进行判决。
- **计算逻辑**：$s = U^T a$ 是对各个“高级特征”的**线性加权**。
  - 例如：如果 $a_1$（探测到 'Museums'）和 $a_2$（探测到 'in'）同时亮起，且 $U$ 给予它们正的权重，那么最终得分 $s$ 就会很高，判定为“是地名”。
- $U$ 是我们在只有两层时给它起的“昵称”，而 $W^{(2)}$ 是它在多层网络家族中的“正式名称”。

------

####  1.4 Maximum Margin Objective Function（最大间隔损失）

这节想要说明：***既然我们已经把网络搭好了（有 $W$ 和 $U$），我们该怎么训练它？** 也就是说，我们需要定义一个“规则”或“指标”（Loss Function），告诉网络什么样的输出是“好”的，什么样的输出是“坏”的。*

不同于让模型直接预测一个绝对数值（比如“这个窗口是地名的概率是 0.8”），采用了一种更直观的对比思路：

- **正样本 ($s$)**：给模型看一句正确的话（比如 "Museums in Paris are amazing"），模型打出的分数为 $s$ 。

- **负样本 ($s_c$)**：给模型看一句被篡改的/错误的话（比如 "Not all museums in Paris"），模型打出的分数为 $s_c$（subscript $c$ 代表 corrupt/损坏） 。

- **目标**：我们希望**真话的得分要明显高于假话的得分**。即希望 $s > s_c$ 。

经典的损失函数公式推导：

- **第一步（朴素想法）**：我们只希望 $s > s_c$。

  - 如果 $s_c > s$，就算误差。
  - 公式：minimize $J = \max(s_c - s, 0)$ 。
  - *问题*：只要 $s$ 比 $s_c$ 大 0.0001，模型就满足了，Loss 变成 0，停止学习。但这太“极限”了，没有容错率。

- **第二步（加入安全边界 Margin）**：

  - 我们希望真话不仅要赢，还要**赢得漂亮**。
  - 真话的得分 $s$ 必须比假话得分 $s_c$ 高出一个固定的**安全距离 $\Delta$**（通常设为 1） 。
  - 如果没有拉开这个距离，就算有 Loss。

- **第三步（最终公式）**：

  $$J = \max(1 + s_c - s, 0)$$

  - 这个函数在机器学习中非常有名，即 **Hinge Loss（铰链损失）**。
  - **含义**：
    - 如果 $s$ 非常高（$s \ge s_c + 1$）：Loss = 0（模型做得很好，甚至超额完成任务，不罚）。
    - 如果 $s$ 不够高（$s < s_c + 1$）：Loss = $1 + s_c - s$（即使 $s > s_c$ 但没超过 1，也要稍微罚一点；如果 $s < s_c$，重罚）。

---

#### 1.5 Training with Backpropagation – Elemental（逐参数的反向传播：链式法则走）

讲义用 toy network 推导一个典型结果：

- $\frac{\partial s}{\partial W^{(1)}_{ij}}=\delta^{(2)}_i\cdot a^{(1)}_j$
- **$\delta$ 是什么？** $\delta$ 是**中间工具（误差信号）**。它代表链式法则中“从输出层传回来的那部分梯度”。并且 $\delta^{(2)}_i$ 表示到达第 2 层第 $i$ 个神经元 pre-activation $z^{(2)}_i$ 的误差信号。

##### Step 0：先写清前向依赖关系

对某一条样本（先忽略 corrupt 那支，只推 $s$ 这一支）：

$z^{(2)} = W^{(1)}a^{(1)}+b^{(1)},\quad a^{(2)} = f(z^{(2)}),\quad s = (W^{(2)})^\top a^{(2)}$

> 其中 $W^{(2)}$ 在讲义里相当于 $U$。
>
> **$U$ 和 $W$ 没有任何区别**。它们都是一个矩阵，作用都是：
>
> 1. 接收上一层的输出（Activation）。
> 2. 进行线性变换（Linear Transformation）。
> 3. 传给下一层。
>
> **$W^{(k)}$ 是把第 $k$ 层输出映射到第 $k+1$ 层输入的转移矩阵** 。

##### **Step 1：从输出往回，对** $W^{(1)}_{ij}$ **用链式法则**

因为 $W^{(1)}_{ij}\to z^{(2)}_i \to a^{(2)}_i \to s$，所以

$$\frac{\partial s}{\partial W^{(1)}_{ij}} =\underbrace{\frac{\partial s}{\partial a^{(2)}_i}\cdot \frac{\partial a^{(2)}_i}{\partial z^{(2)}_i}}_{\text{定义为 } \delta_i^{(2)}} \cdot \underbrace{\frac{\partial z^{(2)}_i}{\partial W^{(1)}_{ij}}}_{\text{Input}}$$

> 分别算三项：
>
> 1. $\displaystyle \frac{\partial s}{\partial a^{(2)}_i}$： $s=\sum_{r} W^{(2)}_{r}a^{(2)}_r\quad \Rightarrow\quad \frac{\partial s}{\partial a^{(2)}_i}=W^{(2)}_{i}$
>
> 2. $\displaystyle \frac{\partial a^{(2)}_i}{\partial z^{(2)}_i}=f'(z^{(2)}_i)$
>
> 3. $\displaystyle \frac{\partial z^{(2)}_i}{\partial W^{(1)}_{ij}}$： 因为 $z^{(2)}_i=b^{(1)}_i+\sum_k a^{(1)}_k W^{(1)}_{ik}$，所以对 $W^{(1)}_{ij}$ 求偏导，只剩第 $k=j$ 项：
>
>    $\frac{\partial z^{(2)}_i}{\partial W^{(1)}_{ij}}=a^{(1)}_j$

所以 $\frac{\partial s}{\partial W^{(1)}_{ij}} = W^{(2)}_i \, f'(z^{(2)}_i)\, a^{(1)}_j$

**定义 $\delta$**：

令 $\delta^{(2)}_i := \frac{\partial s}{\partial z^{(2)}_i} = W^{(2)}_i f'(z^{(2)}_i)$

**结论**：

$$\frac{\partial s}{\partial W^{(1)}_{ij}} = \delta^{(2)}_i \cdot a^{(1)}_j$$

##### Step 2：通解推导（隐层到隐层 / 多路径）

**场景**：第 $k-1$ 层的神经元 $j$ 连接到第 $k$ 层的**所有**神经元（多条路）。

此时，神经元 $j$ 的变动会影响下一层的所有 $z^{(k)}_i$。根据**多元链式法则**，需要把所有路径回传的误差**累加**：

$$\delta^{(k-1)}_j = \frac{\partial s}{\partial z^{(k-1)}_j} = \sum_i \left( \underbrace{\frac{\partial s}{\partial z^{(k)}_i}}_{\delta^{(k)}_i} \cdot \frac{\partial z^{(k)}_i}{\partial z^{(k-1)}_j} \right)$$

其中 $\frac{\partial z^{(k)}_i}{\partial z^{(k-1)}_j} = W^{(k-1)}_{ij} \cdot f'(z^{(k-1)}_j)$。

**结论（通用递归公式）**：

$$\delta^{(k-1)}_j = f'(z^{(k-1)}_j) \cdot \underbrace{\sum_i \delta^{(k)}_i W^{(k-1)}_{ij}}_{\text{Error Sharing: 下一层所有误差的加权和}}$$

##### Step 3：Bias 的梯度

$$z^{(2)}_i = b^{(1)}_i + \sum_k a^{(1)}_k W^{(1)}_{ik} \Rightarrow \frac{\partial z^{(2)}_i}{\partial b^{(1)}_i} = 1$$

所以，Bias 的梯度就是 $\delta$ 本身：

$$\frac{\partial s}{\partial b^{(1)}_i} = \delta^{(2)}_i \cdot 1 = \delta^{(2)}_i$$

*(理解：Bias 可以看作是连接到固定输入“1”的权重，套用 Step 1 公式即得证)*

##### Step 4：把 Hinge Loss 接进来（最容易漏的一步）

上面推的是 $\frac{\partial s}{\partial W}$。真正更新要乘上 $\frac{\partial J}{\partial s}$。

当 $g = 1 + s_c - s > 0$ 时，对 $W$ 求导必须同时考虑它对 $s$（真话）和 $s_c$（假话）的贡献：

$$\frac{\partial J}{\partial s}=-1,\quad \frac{\partial J}{\partial s_c}=+1$$

所以对参数 $W$ 的最终梯度为：

$$\nabla W = \frac{\partial J}{\partial W} = \underbrace{(-1) \cdot \frac{\partial s}{\partial W}}_{\text{让真话得分更高}} + \underbrace{(+1) \cdot \frac{\partial s_c}{\partial W}}_{\text{让假话得分更低}}$$

*(注：当 $g \le 0$ 时，Loss 为 0，梯度为 0，不更新。)*

---

#### 1.6 反向传播（Vectorized）：把“逐元素”变成矩阵公式

1.5 节是为了**理解原理**（显微镜视角：看一个个神经元怎么传误差），那么 1.6 节就是为了**写出高效代码**（工程视角：利用 NumPy/PyTorch 的矩阵加速）。

##### 1. 为什么要向量化？(Motivation: Efficiency)

- **问题**：如果按照 1.5 节的公式直接写代码，需要写多层嵌套的 `for` 循环（遍历层 $k$，遍历神经元 $i$，遍历输入 $j$）。这在 Python 或 MATLAB 等解释型语言中运行速度**极其缓慢** 。
- **解决方案**：科学计算库（如 NumPy）对大规模矩阵乘法（Matrix Multiplication）做了极致优化。**向量化（Vectorized）** 实现通常比循环实现快几个数量级 。

##### 2. 两个核心公式的“矩阵化升级”

###### A. 权重梯度的矩阵化：外积 (Outer Product)

- **标量版 (1.5节)**：对于单个权重 $W_{ij}^{(k)}$，梯度是 $\delta_i^{(k+1)} \cdot a_j^{(k)}$ 。

- **矩阵版 (1.6节)**：可以一次性算出整个权重矩阵 $W^{(k)}$ 的梯度。

  $$\nabla_{W^{(k)}} = \delta^{(k+1)} (a^{(k)})^T$$

  - **解释**：这是下一层的**误差向量** ($\delta^{(k+1)}$) 与当前层的**激活向量** ($a^{(k)}$) 的**外积 (Outer Product)** 。

  - **直觉**：

    $$  \begin{bmatrix} \delta_1 \\ \delta_2 \end{bmatrix}   \times   \begin{bmatrix} a_1 & a_2 & a_3 \end{bmatrix}   =   \begin{bmatrix}   \delta_1 a_1 & \delta_1 a_2 & \delta_1 a_3 \\  \delta_2 a_1 & \delta_2 a_2 & \delta_2 a_3   \end{bmatrix}$$

    这一下就把所有 $W_{ij}$ 的梯度全算出来了，完全消除了循环。

###### B. 误差反向传播的矩阵化：矩阵乘法 (Matrix Multiplication)

- **标量版 (1.5节)**：需要把误差从 $k+1$ 层的所有节点收集回来，再乘上 $f'$。公式里有一个求和符号 $\sum_i \delta_i^{(k+1)} W_{ij}^{(k)}$ 。

- **矩阵版 (1.6节)**：

  $$\delta^{(k)} = f'(z^{(k)}) \circ (W^{(k)T} \delta^{(k+1)})$$

  - **$\circ$ 符号**：代表**逐元素相乘 (Element-wise product / Hadamard product)** 。
  - **$W^{(k)T} \delta^{(k+1)}$**：这一步直接代替了那个求和符号 ($\sum$)。转置矩阵 $W^T$ 正好把梯度的流向反过来了（从输出端流向输入端）。

##### 3. 计算复杂度与复用 (Computational Efficiency)

讲义最后还提到了一个工程细节：**递归复用**。

- $\delta^{(k)}$ 的计算直接依赖于 $\delta^{(k+1)}$ 。
- 所以在反向传播时，算出一层的 $\delta$，就把它存下来（Cache），用来算下一层（实际上是前一层）的 $\delta$。
- 这种递归过程让反向传播在计算上非常“划算”（Affordable）。

---

### 2 Neural Networks: Tips and Tricks（训练技巧）

#### 2.1 Gradient Check（数值梯度检查）

它的核心目的是解决一个非常现实的工程问题：**推导并写代码实现的“反向传播梯度”，到底算得对不对？**

在深度学习中，反向传播的代码很容易写出细微的 bug（比如矩阵维度搞错、正负号弄反、索引错位），而模型依然能跑，只是Loss降不下去。

尽管其计算效率太低,无法直接用于训练网络，但这种方法将使我们能够非常精确地估计关于任何参数的导数。**梯度检查（Gradient Check）** 就是用来给你的反向传播代码检验的工具。

##### 1. 数值梯度的计算公式 (Centered Difference Formula)

讲义介绍了一个比导数标准定义更精确的公式——**中心差分公式（Centered Difference）**：

$$f'(\theta) \approx \frac{J(\theta + \epsilon) - J(\theta - \epsilon)}{2\epsilon}$$

- **操作方法**：
  1. 把要检查的参数 $\theta$ 稍微加一点点（$+\epsilon$，通常 $\epsilon = 10^{-5}$），做一次前向传播，算出 Loss $J(\theta+\epsilon)$。
  2. 把参数 $\theta$ 稍微减一点点（$-\epsilon$），做一次前向传播，算出 Loss $J(\theta-\epsilon)$ 。
  3. 两个 Loss 相减，除以 $2\epsilon$，就是斜率（梯度）。
- **为什么用“双侧”而不是“单侧”？**
  - 标准导数定义是 $f'(x) \approx \frac{f(x+\epsilon) - f(x)}{\epsilon}$。
  - 讲义指出，**中心差分（双侧扰动）** 更精确、更稳定。泰勒展开（Taylor's theorem）证明它的误差是 $O(\epsilon^2)$，而单侧误差是 $O(\epsilon)$ 。

##### 2. 为什么不直接用这个方法训练？(Efficiency vs. Precision)

既然这个方法又准又简单，为什么还要费劲学反向传播？

- **计算量爆炸**：
  - 数值梯度每计算**一个**参数的梯度，就要做**两次**前向传播（一次 $+\epsilon$，一次 $-\epsilon$）。
  - 如果网络有 100 万个参数（这在今天算小的），更新一次权重就需要做 200 万次前向传播。这在计算上是不可接受的（Intractable）。
- **结论**：
  - **数值梯度**：只用于**Debug**。写好代码后，跑一次 Gradient Check，确认没问题了就关掉。
  - **解析梯度（反向传播）**：用于**正式训练**。虽然难写，但算得快。

##### 3. 代码实现细节 (Implementation)

讲义给出了一个 Python (NumPy) 的实现模板（Snippet 2.1）：

- 它遍历输入向量 `x` 的每一个元素（每一个参数）。
- 对每个元素进行 `+h` 和 `-h` 的扰动 。
- 计算 `grad[ix]` 并恢复原来的值 。
- 最后返回计算出的数值梯度向量，你可以把它打印出来和你的反向传播梯度对比。

---

#### 2.2 Regularization（L2 正则）

##### 1. 核心问题：过拟合 (Overfitting / High Variance)

- **现象**：神经网络拥有大量的参数（$W$），这赋予了它极强的拟合能力。如果不加约束，模型容易“死记硬背”训练数据（Training Loss $\to$ 0），导致在未见过的测试数据上表现极差 。

- **目标**：我们希望模型不仅在训练集上表现好，还要具备**泛化能力 (Generalization)**。

  ![overfitting vs underfitting的图片](https://encrypted-tbn3.gstatic.com/licensed-image?q=tbn:ANd9GcTvAabZ8Yrg2e5Zh298VOvJebTMirO2ZyB5Rb7FYhtGlsF7sV5bq5EnU-dN8Wtvk3Yo37v1de2bbDyMal91pDRaGIZxRF017m27-dJgV7ME83z_CYU)

##### **2. 解决方案：$L_2$ 正则化 (L2 Regularization)**

我们在原始损失函数 $J$ 的基础上，强行加上一个“惩罚项”，构成新的总目标函数 $J_R$ ：

$$J_{R} = \underbrace{J}_{\text{原始 Loss}} + \underbrace{\lambda \sum_{i=1}^{L} ||W^{(i)}||_{F}^2}_{\text{正则化惩罚项}}$$

- **$J$ (Original Loss)**：负责**“准确性”**。例如之前的 Max-Margin Loss $J = \max(1 + s_c - s, 0)$。它只在乎预测结果对不对。

- **$||W^{(i)}||_{F}^2$ (Penalty)**：负责**“简单性”**。这是 Frobenius 范数（即矩阵所有元素的平方和） 。

  +1

  - 这里的 $W^{(i)}$ 就是网络中用于前向传播计算 $z=Wx+b$ 的**同一个权重矩阵**。

- **$\lambda$ (Lambda)**：**超参数 (Hyperparameter)**。用于调节“准确”与“简单”之间的权重 。

##### **3. 深度解析：为什么惩罚 $W$ 能防止过拟合？**

- **直觉理解**：
  - 如果 $W$ 中的数值很大，输入 $x$ 的微小变化会被放大，导致输出剧烈波动（函数曲率大，模型过于敏感/复杂）。
  - 通过惩罚 $||W||^2$，迫使权重趋向于 0（即**权重衰减 Weight Decay**）。较小的权重会让模型函数更平滑（Smoother），更难拟合训练数据中的随机噪声 。
- **贝叶斯视角**：这相当于引入了一个**先验信念 (Prior Belief)**，即我们认为最优的参数应该服从以 0 为中心的高斯分布 。

##### **4. 对反向传播的影响 (Gradient Change)**

当优化目标变成 $J_R$ 时，梯度的计算也随之改变：

$$\nabla W_{total} = \frac{\partial J}{\partial W} + \underbrace{2\lambda W}_{\text{衰减项}}$$

- 这意味着每次更新参数时，除了根据误差调整方向，还会把当前的权重“缩小”一点点。

##### **5. 关键细节 (Tips)**

1. **Bias 不参与正则化** ：
   - 公式里只累加了 $W$，没有 $b$。
   - **原因**：偏置 $b$ 只是控制神经元的激活阈值（平移函数），不会增加模型的复杂度（曲率）。惩罚 $b$ 会限制神经元的激活能力，可能导致欠拟合。
2. **$\lambda$ 的调节** ：
   - **$\lambda$ 太大** $\to$ 权重全被压成 0 $\to$ 模型变傻（Underfitting）。
   - **$\lambda$ 太小** $\to$ 约束无效 $\to$ 过拟合（Overfitting）。
   - 必须在验证集（Validation Set）上调优。

------

#### 2.3 Dropout（随机失活）

Dropout 是一种强大的正则化技术,最初由 Srivastava等人在 [Dropout: A Simple WaytoPreventNeural Net- works from  Overfitting](https://dl.acm.org/doi/10.5555/2627435.2670313) 中提出。

##### 1. 核心机制：

- **训练阶段 (Training)**： 在每一次前向传播（Forward Pass）和反向传播中，不是用整个网络，而是**随机“丢弃”（Drop）** 一部分神经元 

  - 具体来说，每个神经元有 $p$ 的概率被保留，有 $(1-p)$ 的概率被置为 0（即被冻结，不参与计算也不更新权重）。
  - 相当于在每一次迭代时，都在训练一个**“残缺”**的子网络。

- **测试阶段 (Testing)**： 不再丢弃任何神经元，而是使用**完整**的网络来进行预测 。

  ![dropout neural network的图片](https://encrypted-tbn3.gstatic.com/licensed-image?q=tbn:ANd9GcQAetOORRBQ4-wH6U4x_QLiG69bIfyMjeG3HRaVh4l8YEa3uTT02m_TrMLE3E73TA-mvr7bNjeXF0eqHLjREzS6W1ZNCReFmW4_XXtG4Aquw5ZWHoI)

##### 2. 直觉：为什么要这么做？(Intuition)

讲义提供了一个非常经典的解释：**集成学习（Ensemble Learning）** 的视角。

- **模型平均**：可以把 Dropout 看作是同时训练了**指数级数量**的“小神经网络” 。
  - 每次由于随机丢弃的模式不同，网络结构都不一样。
  - 最后测试时使用全网络，相当于把这些成千上万个小网络的预测结果取了一个**平均**。
- **防止依赖**：这迫使神经元不能过度依赖某些特定的邻居神经元（因为邻居随时可能“消失”）。它必须学会独立提取更有鲁棒性的特征。

##### 3. 关键细节：缩放问题 (The Scaling Subtlety)

面试中关于 Dropout **最常考**的数学细节，讲义专门提到了这一点 。

- **问题**：
  - 在**训练**时，只有 $p$ 比例的神经元是活着的。假设每个神经元输出 1，那么总输出期望是 $p \times 1 + (1-p) \times 0 = p$。
  - 在**测试**时，所有神经元都活着。总输出期望变成了 $1$。
  - **结果**：测试时的激活值（Activation）会比训练时**大很多**（大了 $1/p$ 倍），导致网络彻底乱套 。
- **解决方案**： 我们需要保证**期望输出一致**。通常有两种做法：
  1. **测试时缩放**：在测试时，把所有权重乘以 $p$。
  2. **反向 Dropout (Inverted Dropout)**（这是现在 PyTorch/TensorFlow 的默认做法）：在**训练时**，就把激活值除以 $p$（即放大 $1/p$ 倍）。这样测试时就不用动了，直接用全网络即可。

---

#### 2.4 其它激活函数

在之前的章节中，一直默认使用 **Sigmoid** 函数来引入非线性 。但在实际应用中，其他激活函数往往能设计出更好的网络 

##### 1. Sigmoid (经典但老旧)

- **定义**：这是我们之前一直在用的默认选择 。

  $$\sigma(z) = \frac{1}{1+\exp(-z)}$$

- **范围**：$(0, 1)$ 。

- **导数**：$\sigma'(z) = \sigma(z)(1-\sigma(z))$ 。

- 它的输出不是以 0 为中心的（全是正数），这会导致梯度更新时出现“锯齿状”路径，收敛较慢。且在两端（饱和区）梯度几乎为 0，导致**梯度消**

##### 2. Tanh (双曲正切)

- **定义**：它是 Sigmoid 的缩放变体。公式为 $\tanh(z) = \frac{\exp(z)-\exp(-z)}{\exp(z)+\exp(-z)} = 2\sigma(2z) - 1$ 。
- **范围**：输出值在 $(-1, 1)$ 之间 。
- **优势**：它的输出是以 **0 为中心 (Zero-centered)** 的（有正有负）。这解决了 Sigmoid 输出恒正的问题，是循环神经网络（RNN/LSTM）中的首选。

- **导数**：$\tanh'(z) = 1 - \tanh^2(z)$ 。

##### 3. Hard Tanh (硬双曲正切)

- **定义**：这是一个分段线性函数。当 $z > 1$ 时输出 1，当 $z < -1$ 时输出 -1，中间是线性 $z$ 。
- **优势**：**计算成本更低** (computationally cheaper) 。因为算指数函数 $\exp(z)$ 比较费时，直接比较大小（Clip）非常快。
- **劣势**：当 $|z| > [cite_start]1$ 时，梯度直接变成 0（饱和），完全停止学习 。

##### 4. Soft Sign

- **定义**：$softsign(z) = \frac{z}{1+|z|}$ 。
- **优势**：它是 Tanh 的另一种替代品。相比 Hard Tanh，它**不那么容易饱和** (does not saturate as easily as hard clipped functions) 。它在两端趋向于 $\pm 1$ 的速度比 Tanh 慢，也就是梯度消失得慢一些。

##### 5. ReLU (线性整流单元) —— **重点**

这是现代深度学习（特别是计算机视觉）中最流行的激活函数 。

- **定义**：$rect(z) = \max(z, 0)$ 。

- **优势**：

  1. **不饱和**：对于所有正数输入 ($z > 0$)，它完全不饱和（梯度恒为 1）。这意味着无论输入多大，梯度都能顺畅地传回去，极大地缓解了梯度消失问题。
  2. **计算快**：只需要判断是否大于 0，没有任何指数运算。

- **劣势**：Dead ReLU 问题。如果输入是负数，输出是 0，梯度也是 0 。一旦神经元陷入这个状态，它就再也不会更新了。

- **导数**：

  $$  \text{rect}'(z) = \begin{cases} 1 & : z > 0 \\ 0 & : \text{otherwise} \end{cases}$$

##### 6. Leaky ReLU (带泄露的 ReLU)

- **动机**：为了解决 ReLU 的一个缺陷——**Dead ReLU Problem**。 标准的 ReLU 在 $z < 0$ 时梯度为 0 。如果一个神经元的输入始终为负（比如初始化不好或学习率太大），它就“死”了，再也无法更新参数。

- **定义**：为了解决 ReLU 在负区间“死亡”的问题，Leaky ReLU 允许负数区域有一个很小的斜率 $k$。其中 $0 < k < 1$（$k$ 是一个小常数，如 0.01） 

  $$leaky(z) = \max(z, k \cdot z)$$

- **机制**：这样即使 $z$ 是负的，也会有一个小的梯度 $k$ 传回去 ，保证神经元总是能学到一点东西。

- **导数**：

  $$  \text{leaky}'(z) = \begin{cases} 1 & : z > 0 \\ k & : \text{otherwise} \end{cases}$$

---

这份笔记融合了讲义核心内容、我们刚才的讨论（直觉与类比）、以及面试中常考但容易被忽略的“隐形问题”。

你可以直接将这份笔记加入你的复习资料库。

------

#### **2.5 Data Preprocessing（数据预处理：为优化器铺路）**

**核心目的**：

模型无法理解数据的物理单位（比如“米” vs “毫米”）。如果直接喂入原始数据，不同特征的数值范围差异巨大，会导致 Loss 的等高线变成狭长的椭圆，梯度下降路线震荡，难以收敛。预处理就是要把数据“整形”成优化器喜欢的样子（圆形等高线）。

##### **1. 常用技术**

- **Mean Subtraction（去均值 / 中心化）** 
  - **操作**：$X' = X - \mu$。
  - **目的**：将数据整体平移，使其以 0 为中心（Zero-centered）。
  - **关键细节（面试考点）**：均值 $\mu$ **只能在训练集（Training Set）上计算** 。然后用这个训练集的 $\mu$ 去处理验证集和测试集。
  - *为什么？* 如果你用整个数据集算均值，相当于让测试集的信息“泄露”给了模型（Data Leakage），导致评估结果虚高。
- **Normalization（归一化 / 标准化）** 
  - **操作**：$X'' = X' / \sigma$ （除以标准差）。
  - **目的**：解决特征量纲不一致的问题。让所有特征都在相似的尺度范围内（通常是 -1 到 1 左右）。
  - *直觉*：防止数值大的特征（如房屋面积 100-200）掩盖了数值小的特征（如卧室数 2-5）的梯度贡献。
- **Whitening（白化）** 
  - **操作**：在去均值的基础上，通过 SVD（奇异值分解）将数据投影到特征基上，并除以奇异值。
  - **目的**：消除特征之间的**相关性（Correlation）**，使协方差矩阵变为单位矩阵 。
  - *现状*：虽然统计学上很美，但在深度学习中计算成本高，**不如前两者常用** 。

------

#### 2.6 Parameter Initialization（参数初始化：打破对称与保持方差）

**核心目的**：

决定了网络能否“启动”。初始权重 $W$ 就像多米诺骨牌的第一张，如果推得太轻（权重太小）或太重（权重太大），信号传到后面就断了（梯度消失/爆炸）。

##### **1. 错误的初始化方式（面试问）**

- **全零初始化 (Zero Initialization)**：
  - **后果**：**对称性破坏失败 (Symmetry Breaking Fail)**。
  - *解释*：如果 $W=0$，所有神经元接收到的输入一样，输出一样，反向传播算的梯度也一样。无论训练多久，所有神经元都在学完全相同的东西，网络退化成线性模型。
- **过小/过大的随机初始化**：
  - **过小**：信号在层层传递中不断乘以小于 1 的数 $\to$ **梯度消失 (Vanishing Gradient)** 。
  - **过大**：信号不断放大 $\to$ **梯度爆炸**，或者导致 Sigmoid/Tanh 进入饱和区（导数为 0） $\to$ 梯度消失。

##### **2. Xavier (Glorot) Initialization**

- **核心思想**：**方差保持 (Variance Preservation)** 。

  - 前向传播时，输出的方差 $\approx$ 输入的方差。
  - 反向传播时，梯度的方差 $\approx$ 下一层梯度的方差。

- **公式**：

  从均匀分布中采样 $W$：

  $$W \sim U \left[ -\sqrt{\frac{6}{n_{in} + n_{out}}}, +\sqrt{\frac{6}{n_{in} + n_{out}}} \right]$$

  - $n_{in}$：输入单元数 (Fan-in)。
  - $n_{out}$：输出单元数 (Fan-out)。
  - *直觉*：权重的大小应该与层宽（节点数）的平方根成**反比**。

- **适用范围**：最适合 **Sigmoid** 和 **Tanh** 激活函数 。

##### **3. 问题（Deep Dive）**

- **Q1: Bias (偏置) 需要 Xavier 初始化吗？**
  - **不需要。** 讲义明确指出，Bias 通常初始化为 **0** 。
  - *原因*：Bias 只是平移激活函数，不参与“乘法”运算，不会导致信号的方差被放大或缩小。如果强行设置 Bias，反而可能引入不必要的先验偏置。
- **Q2: 如果用 ReLU 激活函数（现在的标准做法），还能用 Xavier 吗？**
  - **讲义未提，但实战关键**：Xavier 假设激活函数是线性的（在 0 附近）。但 ReLU 会把一半的输入（负数）直接砍成 0，导致方差**减半**。
  - 如果用 Xavier 初始化 ReLU 网络，深层网络的信号会越来越弱。
  - **解决方案**：使用 **He Initialization (Kaiming Init)**。它的公式里系数是 $\sqrt{2/n_{in}}$（乘了一个 2 来补偿 ReLU 砍掉的一半方差）。

---

这三节（2.7 Learning Strategies, 2.8 Momentum Updates, 2.9 Adaptive Optimization Methods）进入了深度学习中最具**“炼丹”**色彩的领域：**优化器（Optimizers）**。

如果说前面的章节是“设计了一辆跑车（网络结构）”并“加好了油（数据预处理）”，那么这三节就是在讲**“如何踩油门”**（调整学习率和更新策略），才能让这辆车最快、最稳地到达终点（Loss 最低点）。

这部分内容的演进史，就是从**手动挡（SGD）\**进化到\**自动挡（Adam）**的历史。

------

#### **2.7 Learning Strategies（学习率策略：如何踩油门）**

这一节讨论了一个最让初学者头疼的超参数：**学习率（Learning Rate, $\alpha$）**。它决定了参数每次更新的步长大小 。

##### **1. 学习率**

- **太大 ($\alpha$ too big)**： 模型会“超速”。它可能直接越过最低点，甚至导致 Loss **发散 (Diverge)** 。就像图 15 展示的那样，本来要在谷底停车，结果冲到了对面的山上
- **太小 ($\alpha$ too small)**：模型会“龟速”爬行。虽然能保证不发散，但收敛时间极长，或者容易卡在**局部极小值 (Local Minima)** 动弹不得 。
- **刚刚好**：你需要精细调整（Tune）这个参数 。

##### **2. 解决方案：学习率衰减 (Annealing)**

既然很难选一个固定的 $\alpha$，那我们能不能**“先大后小”**？

- **原理**：刚开始训练时，离目标很远，步子大一点（High Learning Rate）以便快速靠近；当接近谷底时，步子小一点，进行微调（Fine-grained scope） 。

- **常见策略**：
1. **Step Decay**：每隔 $n$ 轮（Epochs），把学习率砍半（比如 $\times 0.5$） 。
  
2. **Exponential Decay**：按指数衰减 $\alpha(t) = \alpha_0 e^{-kt}$ 。
  
3. **1/t Decay**：$\alpha(t) = \frac{\alpha_0 \tau}{\max(t, \tau)}$ 。

> **⚡️ 当前技术对比 (Modern Practice)**：
>
> 虽然讲义提到的衰减策略（Decay）现在还在用，但现代大模型训练（如 BERT, GPT, LLaMA）更常用 **"Warmup + Cosine Decay"**。
>
> - **Warmup**：刚开始不是直接用大学习率，而是从 0 慢慢爬升到设定的 $\alpha$。这是为了防止模型在初始化阶段（梯度还不稳定时）因为步长太大而崩掉。
> - **Cosine Decay**：之后按余弦曲线平滑下降，比阶梯式下降（Step Decay）更顺滑，效果通常更好。

------

#### **2.8 Momentum Updates（动量更新：给 SGD 加个惯性）**

这一节引入了物理学中的**“动量”**概念。

标准的 SGD 有个缺点：它只看当前的梯度。如果地形是狭长的山谷（Ravine），SGD 会在山谷两壁之间来回震荡，前进缓慢。

##### **1. 核心思想**

如果不只是看当前的“坡度”（梯度），而是保留一部分**之前的“速度”**呢？

- **公式**：

  $$v = \mu v - \alpha \nabla J$$

  $$\theta = \theta + v$$

  这里 $v$ 是速度，$\mu$ 是摩擦系数（或衰减率，通常是 0.9），$\nabla J$ 是当前梯度 。

- **直觉**：就像一个球滚下山坡。即使当前梯度突然变了方向（震荡），球因为有**惯性（Momentum）**，依然会保持原来的大概方向继续冲。这能加速收敛并抑制震荡 。

> **⚡️ 当前技术对比**：
>
> 在现在的 PyTorch/TensorFlow 中，当调用 `torch.optim.SGD` 时，通常都会把 `momentum` 参数设为 **0.9**。纯粹的不带动量的 SGD 在深度学习中已经很少使用了。

------

#### **2.9 Adaptive Optimization Methods（自适应优化：智能自动挡）**

这是优化器进化的终极形态。这一节解决的问题是：**前面的方法对所有参数都用同一个学习率，这合理吗？** 有些特征（比如 NLP 中的罕见词）出现频率很低，梯度很稀疏；有些特征出现频率很高。我们希望：**对罕见特征更新猛一点，对常见特征更新稳一点** 。

讲义介绍了三个里程碑式的算法：

##### **1. AdaGrad (Adaptive Gradient)**

- **机制**：给每个参数分配独立的学习率。

- **公式**：$\theta_{t,i} = \theta_{t-1,i} - \frac{\alpha}{\sqrt{\sum g^2}} g_{t,i}$ 。

- **特点**：分母是**历史梯度的平方和**。如果某个参数以前更新了很多次（梯度大），分母就大，学习率自动变小。

- **缺陷**：分母是**单调递增**的（一直在累加正数）。训练到后期，学习率会变得无穷小，模型提前停止学习 。

##### **2. RMSProp (Root Mean Square Propagation)**

- **改进**：解决了 AdaGrad 学习率过早消失的问题。

- **机制**：不再简单累加所有历史梯度，而是计算**滑动平均（Moving Average）**。

  `cache = decay_rate * cache + (1 - decay_rate) * dx**2` 。

- **效果**：它“遗忘”了太久远的梯度历史，只关注最近的梯度震荡情况。

##### 3. Adam (Adaptive Moment Estimation) 

- **地位**：这是目前几乎所有生成式模型（包括你的 TrafficLISA 和 FlowDiffGAN）的**默认首选**。

- **原理**：它是 **Momentum + RMSProp** 的合体 。

  - 它像 Momentum 一样，维护一个梯度的**一阶矩**（均值，代码里的 `m`） 。

  - 它像 RMSProp 一样，维护一个梯度的**二阶矩**（方差，代码里的 `v`） 。

- **公式**：

  $$m = \beta_1 m + (1-\beta_1) dx$$

  $$v = \beta_2 v + (1-\beta_2) dx^2$$

  $$x += - \text{learning\_rate} \cdot \frac{m}{\sqrt{v} + \epsilon}$$

> **⚡️ 当前技术对比 (Adam vs. AdamW)**：
>
> 虽然讲义里讲的是 Adam，但现在大家用的其实是 **AdamW**（PyTorch 中的 `torch.optim.AdamW`）。
>
> - **Bug 修复**：原始的 Adam 在处理 L2 正则化（Weight Decay）时有一个数学上的实现细节偏差。
> - **AdamW**：把 Weight Decay 从梯度更新中解耦出来，这对于 Transformer 等大模型的泛化能力至关重要。
> - **显存优化**：对于超大模型，现在还有 **8-bit Adam**（把优化器状态量化以节省显存）。

------





