前置内容：LSTM的6个通用公式：

> 1. 输入门：
>
>    $i_t = \sigma(W^{(i)}x_t + U^{(i)}h_{t-1})$
>
> 2. 遗忘门：
>
>    $f_t = \sigma(W^{(f)}x_t + U^{(f)}h_{t-1})$
>
> 3. 新的候选记忆：
>
>    $\tilde{c}_t = \tanh(W^{(c)}x_t + U^{(c)}h_{t-1})$
>
> 4. 最终记忆细胞：
>
>    $c_t = f_t \circ c_{t-1} + i_t \circ \tilde{c}_t$
>
> 5. 输出们和最终状态：
>
>    $o_t = \sigma(W^{(o)}x_t + U^{(o)}h_{t-1})$
>
>    $h_t = o_t \circ \tanh(c_t)$

<img src="/Users/yyy/Library/Application Support/typora-user-images/image-20260223135824820.png" alt="image-20260223135824820" style="zoom:50%;" />

整体流程：

> **编码器端**：源语言经过 Embedding 和 CNN 层后，送入双向 LSTM，生成了包含全局上下文的编码器隐藏状态 $h_i^{enc}$。
>
> **解码器端 (时间步 $t$)**：解码器 LSTM 基于上一步状态和当前输入，生成了当前的隐藏状态 $h_t^{dec}$。
>
> **注意力融合**：利用 $h_t^{dec}$ 去“查询”所有的 $h_i^{enc}$，计算出注意力输出 $a_t$。
>
> **组合输出**：将 $a_t$ 和 $h_t^{dec}$ 拼接，通过线性层 $W_u$、$\tanh$ 激活函数和 Dropout，得到了当前时间步的**最终特征表示 $o_t$**。
>
> 现在我们拿到了当前时刻最高级的特征 $o_t$，我们把它映射到目标词表的维度（$V_t$）：
>
> $z_t^{(out)} = W_{vocab} o_t$。（这里 $W_{vocab}$ 就是最后一层的权重矩阵，$o_t$ 相当于上一层的激活输出）。
>
> 将 Logits 转化为概率分布：$P_t = \text{softmax}(z_t^{(out)}) = \text{softmax}(W_{vocab} o_t)$
>
> 损失计算 ：$J_t(\theta) = \text{CrossEntropy}(P_t, g_t)$ 这里的 $g_t$ 就是真实单词的 One-hot 向量。

---

代码实现与通用公式的详细说明：

第一阶段：编码源句子 (Encoder)

1. 词嵌入与卷积 (Embedding & CNN Layer)

   - 将源语言句子中的每个字符转化为词嵌入向量 $\mathbf{x}_i \in \mathbb{R}^{e \times 1}$
   - 对应图中底部的黄色梯形 `CNN Layer`。它在嵌入向量上滑动，提取局部特征（例如将“电”和“脑”结合成“电脑”的特征），输出维度保持不变。

   ```python
   """
   - `vocab.src` 和 `vocab.tgt` 分别是源语言和目标语言的 `VocabEntry` 对象
   - `len(vocab.src)` 给出源语言词汇量大小，`len(vocab.tgt)` 给出目标语言词汇量
   - `src_pad_token_idx` 和 `tgt_pad_token_idx` 已经帮你算好了（都是 0）
   - `self.embed_size` 是嵌入维度
   """
   # 初始化源语言嵌入层
   self.source = 
   nn.Embedding(len(vocab.src), self.embed_size, padding_idx=src_pad_token_idx)
   # 初始化目标语言嵌入层
   self.target = nn.Embedding(len(vocab.tgt),self.embed_size,padding_idx=tgt_pad_token_idx)
   ```

   初始化 NMT 模型的 9 个组件。这里要特别注意每一层的**输入/输出维度**。

   > 用 `e` 表示 `embed_size`，`h` 表示 `hidden_size`，`V_t` 表示 `len(vocab.tgt)`

   | 变量                              | 层类型        | 关键参数                                                     |
   | --------------------------------- | ------------- | ------------------------------------------------------------ |
   | `self.post_embed_cnn`             | `nn.Conv1d`   | `in_channels=e`, `out_channels=e`, `kernel_size=2`, `padding='same'` |
   | `self.encoder`                    | `nn.LSTM`     | `input_size=e`, `hidden_size=h`, `bidirectional=True`, `bias=True` |
   | `self.decoder`                    | `nn.LSTMCell` | `input_size=e+h`, `hidden_size=h`, `bias=True`               |
   | `self.h_projection`               | `nn.Linear`   | `in_features=2h`, `out_features=h`, `bias=False`             |
   | `self.c_projection`               | `nn.Linear`   | `in_features=2h`, `out_features=h`, `bias=False`             |
   | `self.att_projection`             | `nn.Linear`   | `in_features=2h`, `out_features=h`, `bias=False`             |
   | `self.combined_output_projection` | `nn.Linear`   | `in_features=3h`, `out_features=h`, `bias=False`             |
   | `self.target_vocab_projection`    | `nn.Linear`   | `in_features=h`, `out_features=V_t`, `bias=False`            |
   | `self.dropout`                    | `nn.Dropout`  | `p=self.dropout_rate`                                        |

    > **为什么 decoder 的 input_size 是 `e+h`**：因为每步输入 $\bar{y}_t$ 是目标嵌入 $y_t \in \mathbb{R}^e$ 和上一步 combined-output $o_{t-1} \in \mathbb{R}^h$ 的拼接。
   >
   > **为什么 combined_output_projection 是 `3h`**：因为输入 $u_t = [a_t; h_t^{dec}]$，其中 $a_t \in \mathbb{R}^{2h}$ 和 $h_t^{dec} \in \mathbb{R}^h$。



2. 双向 LSTM 编码 (Bidirectional LSTM)

   - 将 CNN 的输出送入双向 LSTM 中。前向 LSTM（从左到右阅读）和后向 LSTM（从右到左阅读）分别提取特征。

   - 隐藏状态：$\mathbf{h}_i^{enc} = [\overrightarrow{\mathbf{h}_i^{enc}}; \overleftarrow{\mathbf{h}_i^{enc}}]$

   - 细胞状态：$\mathbf{c}_i^{enc} = [\overrightarrow{\mathbf{c}_i^{enc}}; \overleftarrow{\mathbf{c}_i^{enc}}]$

     - $\mathbf{h}_i^{enc} = [\overrightarrow{\mathbf{h}_i^{enc}}; \overleftarrow{\mathbf{h}_i^{enc}}]$就是把**两套独立运转的 LSTM 引擎（各跑了 6 个公式）的最终输出 $h_t$**，像拼积木一样左右拼在了一起。

   - 对应图中红色的垂直方块（包含4个红色圆圈）。由于是前向和后向的拼接，所以输出的维度翻倍，变为 $\mathbb{R}^{2h \times 1}$。这些 $\mathbf{h}_i^{enc}$ 将作为后续注意力机制的“记忆库”。

     ```python
     # 构造源语言句子张量x, shape (src_len, b, e)
     X = self.model_embeddings.source(source_padded)
     
     # 将x的形状从 (src_len, b, e) 变为 Conv1d 期望输入形状 (batch, channels, length)，即 (b, e, src_len)
     X = X.permute(1, 2, 0)
     
     # 应用后嵌入 CNN 层
     X = self.post_embed_cnn(X)
     
     # 将x的形状从 (b, e, src_len) 变为 (src_len, b, e)
     X = X.permute(2, 0, 1)
     
     # 应用编码器
     packed_X = pack_padded_sequence(X, source_lengths, batch_first= False)
     # 应用编码器  packed_X 是 PackedSequence 对象，包含输入序列和长度信息
     enc_hiddens, (last_hidden, last_cell) = self.encoder(packed_X)
     # batch_first=True 使得 pad_packed_sequence 的输出直接就是 (b, src_len, h*2)。
     enc_hiddens = pad_packed_sequence(enc_hiddens, batch_first= True)[0]
     
     ```

     

第二阶段：桥接初始化 (Initialization Bridge)

- 由于编码器是双向的（维度 $2h$），而解码器是单向的（维度 $h$），所以不能直接赋值。必须通过一个线性投影层 $\mathbf{W}_h$ 和 $\mathbf{W}_c$ 将其降维，作为解码器的初始状态。

  - $\mathbf{h}_0^{dec} = \mathbf{W}_h [\overleftarrow{\mathbf{h}_1^{enc}}; \overrightarrow{\mathbf{h}_m^{enc}}]$
  - $\mathbf{c}_0^{dec} = \mathbf{W}_c [\overleftarrow{\mathbf{c}_1^{enc}}; \overrightarrow{\mathbf{c}_m^{enc}}]$
  - 注意这里提取的是前向 LSTM 的**最后一个**状态 $\overrightarrow{\mathbf{h}_m^{enc}}$ 和后向 LSTM 的**第一个**状态 $\overleftarrow{\mathbf{h}_1^{enc}}$，把它们拼接起来，就包含了整个句子的最全息的摘要。

  ```python
  # 计算解码器初始状态
  # 将双向编码器的最终隐状态和细胞状态拼接后投影，作为解码器的初始状态
  init_decoder_hidden = torch.cat([last_hidden[0], last_hidden[1]], dim=1)
  init_decoder_hidden = self.h_projection(init_decoder_hidden)
  
  # 将双向编码器的最终细胞状态拼接后投影，作为解码器的初始细胞状态
  init_decoder_cell = torch.cat([last_cell[0], last_cell[1]], dim=1)
  init_decoder_cell = self.c_projection(init_decoder_cell)
  
  # 将解码器的初始状态和细胞状态打包成一个元组
  dec_init_state = (init_decoder_hidden, init_decoder_cell)
  ```

  

第三阶段：单步解码与注意力聚焦 (Decoder & Attention)

1. 初始化：

   ```python
   # decode方法中
   # 应用注意力投影层 将编码器隐藏状态投影到注意力空间 (b, src_len, 2h) 变成 (b, src_len, h)
   enc_hiddens_proj = self.att_projection(enc_hiddens)
   
   # 构造目标语言句子张量y, shape (tgt_len, b, e)
   # target_padded 形状 (tgt_len-1, b) → Y 形状 (tgt_len-1, b, e)
   Y = self.model_embeddings.target(target_padded)
   
   # 使用 torch.split 将 Y 沿着时间维度拆分成 tgt_len-1 个张量，每个张量形状 (1, b, e)
   Y_t = torch.split(Y, 1, dim=0)
   # 将 Y_t 中的每个张量 squeeze 掉维度 0，得到形状 (b, e) 的张量
   Y_t = [y.squeeze(0) for y in Y_t]
   ```

   

- 假设我们现在处于第 $t$ 步，已经翻译出了 `<START> I like`，正准备预测下一个词 `desserts`。

1. 解码器前进一步 (Decoder Step)

   - 将当前目标词的嵌入 $\mathbf{y}_t$（例如 "like" 的词向量）与**上一步的组合输出** $\mathbf{o}_{t-1}$ 拼接，组成 $\overline{\mathbf{y}}_t$。将其送入解码器 LSTM。
     - 同一个 \(\mathbf{o}_{t}\) 被用于两个用途：预测当前词$$\mathbf{P}_t = \text{softmax}(\mathbf{W}_{vocab} \mathbf{o}_t)$$ ; 反馈给下一步输入 $\overline{\mathbf{y}}_t = [\mathbf{y}_t;\; \mathbf{o}_{t-1}] \in \mathbb{R}^{(e+h) \times 1}$
   - $\mathbf{h}_t^{dec}, \mathbf{c}_t^{dec} = \text{Decoder}(\overline{\mathbf{y}}_t, \mathbf{h}_{t-1}^{dec}, \mathbf{c}_{t-1}^{dec})$
   - 对应图中绿色的垂直方块，输出当前的“内部想法” $\mathbf{h}_t^{dec}$。

   ```python
   # 遍历 Y_t，构造 Ybar_t
   # Ybar_t 形状 (b, e + h)
   for y_t in Y_t:
       # 将 y_t 和 o_prev 拼接，得到形状 (b, e + h) 的张量
       Ybar_t = torch.cat([y_t, o_prev], dim=-1)
       # 应用解码器 step 函数，得到新的解码器状态 (dec_hidden, dec_cell) 形状 (b, h) 和新的组合输出 o_t 形状 (b, h)    
       dec_state, o_t, _ = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
       # 将 o_t 添加到 combined_outputs 列表中
       combined_outputs.append(o_t)
       # 更新 o_prev 为新的 o_t 形状 (b, h)
       o_prev = o_t
   
   # 将 combined_outputs 列表转换为形状 (tgt_len, b, h) 的张量
   combined_outputs = torch.stack(combined_outputs)
   
   ```

   ```python
   # step方法中
   # 应用解码器 step 函数，得到新的解码器状态 (dec_hidden, dec_cell) 形状 (b, h) 和新的组合输出 o_t 形状 (b, h)
   dec_state = self.decoder(Ybar_t, dec_state)
   dec_hidden, dec_cell = dec_state
   
   ```

   

2. 乘性注意力机制 (Multiplicative Attention)

   - **动作 A（计算注意力分数）**：用解码器当前状态 $\mathbf{h}_t^{dec}$ 去和所有的编码器状态 $\mathbf{h}_i^{enc}$ 算内积得分。

     - $\mathbf{e}_{t,i} = (\mathbf{h}_t^{dec})^T \mathbf{W}_{attProj} \mathbf{h}_i^{enc}$
     - 图中绿框向左上角发出的连线，经过蓝色圆圈（点积/投影操作），算出 $e_t$。

     ```python
     # 计算注意力得分 e_t 形状 (b, src_len)
     # 结果形状：(b, src_len, h) @ (b, h, 1) → (b, src_len, 1) → squeeze → (b, src_len)
     e_t = torch.bmm(enc_hiddens_proj, dec_hidden.unsqueeze(2)).squeeze(2)
     
     # Set e_t to -inf where enc_masks has 1
     # 如果 enc_masks 不为 None，则将 e_t 中对应位置的值设为 -inf
     if enc_masks is not None:
         e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))
     ```

   - **动作 B（计算注意力分布）**：将分数转化为总和为 1 的概率分布。

     - $\alpha_t = \text{softmax}(\mathbf{e}_t)$
     - 图中浅蓝色的柱状图。柱子越高，代表模型当前越关注对应的中文词（比如此时“甜点”对应的柱子最高）。

   - **动作 C（提取上下文向量）**：根据权重对编码器状态进行加权求和。

     - $\mathbf{a}_t = \sum \alpha_{t,i} \mathbf{h}_i^{enc}$
     - 虚线汇聚到上方，生成红色的 `Attention Output` $\mathbf{a}_t$。

3. 合成最终特征 (Combined Output)

   - 将解码器自身的想法 $\mathbf{h}_t^{dec}$ 和通过注意力看原文得到的重点 $\mathbf{a}_t$ 融合起来，经过线性降维、$\tanh$ 激活和 Dropout 处理。

     - 拼接：$\mathbf{u}_t = [\mathbf{a}_t; \mathbf{h}_t^{dec}]$
     - 降维：$\mathbf{v}_t = \mathbf{W}_u \mathbf{u}_t$
     - 激活：$\mathbf{o}_t = \text{dropout}(\tanh(\mathbf{v}_t))$

   - 图中最右侧，红色的 $\mathbf{a}_t$ 和绿色的 $\mathbf{h}_t^{dec}$ 汇聚到那个蓝色小圆圈中，最终生成了 `Combined Output Vector` $\mathbf{o}_t$

     ```python
     # 应用 softmax 函数，得到注意力权重 alpha_t 形状 (b, src_len)
     alpha_t = F.softmax(e_t, dim=1)
     
     # 结果形状：(b, src_len, h) @ (b, src_len, 2h) → (b, 1, 2h) → squeeze → (b, 2h)
     a_t = torch.bmm(alpha_t.unsqueeze(1), enc_hiddens).squeeze(1)
     
     # 拼接u_t, 形状 (b, 3h)
     u_t = torch.cat([a_t, dec_hidden], dim=1)
     
     # 投影v_t, 形状 (b, h)
     v_t = self.combined_output_projection(u_t)
     
     # 应用 Tanh 函数，得到 O_t 形状 (b, h)
     O_t = torch.tanh(v_t)
     
     # 应用丢弃层，得到 combined_output 形状 (b, h)
     O_t = self.dropout(O_t)
     ```

第四阶段：预测与计算损失 (Prediction & Loss)

1. 预测分布 (Output Distribution)
   - 将 $\mathbf{o}_t$ 映射到目标语言词表大小，并转化为概率分布。
     - $\mathbf{P}_t = \text{softmax}(\mathbf{W}_{vocab} \mathbf{o}_t)$
   - 图右上角的蓝色柱状图，代表词表中每个词作为下一个词的概率。模型在这里高置信度地预测了 `desserts`。
2. 计算损失 (Cross Entropy Loss)
   - 对比模型预测的概率分布 $\mathbf{P}_t$ 和真实的单词标签 $\mathbf{g}_t$（One-hot 向量），计算交叉熵损失。
     - $J_t(\theta) = \text{CrossEntropy}(\mathbf{P}_t, \mathbf{g}_t)$

```python
###     Run the network forward:
###     1. Apply the encoder to `source_padded` by calling `self.encode()`
###     2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
###     3. Apply the decoder to compute combined-output by calling `self.decode()`
###     4. Compute log probability distribution over the target vocabulary using the
###        combined_outputs returned by the `self.decode()` function.

enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

# Zero out, probabilities for which we have nothing in the target text
target_masks = (target_padded != self.vocab.tgt['<pad>']).float()

# Compute log probability of generating true target words
target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(
    -1) * target_masks[1:]
scores = target_gold_words_log_prob.sum(dim=0)
```

---

所有关键变量的维度（`b` = batch, `m` = src_len, `e` = embed_size, `h` = hidden_size）：

| 变量               | 公式对应                                                     | 形状                    | 说明                 |
| ------------------ | ------------------------------------------------------------ | ----------------------- | -------------------- |
| `source_padded`    |                                                              | `(m, b)`                | 填充后的源句子       |
| `X` (embedding 后) | $\mathbf{x}_i$                                               | `(m, b, e)`             | 源嵌入               |
| `enc_hiddens`      | $\mathbf{h}_i^{enc}$                                         | `(b, m, 2h)`            | 编码器隐状态         |
| `last_hidden`      | $\overrightarrow{\mathbf{h}_m^{enc}}, \overleftarrow{\mathbf{h}_1^{enc}}$ | `(2, b, h)`             | 双向 LSTM 最后隐状态 |
| `dec_init_state`   | $\mathbf{h}_0^{dec}, \mathbf{c}_0^{dec}$                     | `tuple: (b, h), (b, h)` | 解码器初始状态       |
| `enc_hiddens_proj` | $\mathbf{W}_{attProj}\mathbf{h}_i^{enc}$                     | `(b, m, h)`             | 注意力投影后         |
| `Y`                | $\mathbf{y}_t$                                               | `(tgt_len, b, e)`       | 目标嵌入             |
| `Ybar_t`           | $\overline{\mathbf{y}}_t = [\mathbf{y}_t; \mathbf{o}_{t-1}]$ | `(b, e+h)`              | 每步解码器输入       |
| `dec_hidden`       | $\mathbf{h}_t^{dec}$                                         | `(b, h)`                | 解码器隐状态         |
| `e_t`              | $\mathbf{e}_t$                                               | `(b, m)`                | 注意力分数           |
| `alpha_t`          | $\alpha_t$                                                   | `(b, m)`                | 注意力权重           |
| `a_t`              | $\mathbf{a}_t$                                               | `(b, 2h)`               | 注意力输出           |
| `u_t`              | $\mathbf{u}_t$                                               | `(b, 3h)`               | 拼接向量             |
| `o_t`              | $\mathbf{o}_t$                                               | `(b, h)`                | combined output      |

---



