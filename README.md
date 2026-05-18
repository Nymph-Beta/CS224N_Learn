# CS224N: Natural Language Processing with Deep Learning

Stanford CS224N 课程学习记录，包含 assignment 代码、课程笔记，以及手写 Transformer
核心结构的复现练习。

## 学习进度

| Assignment | 主题 | 状态 |
|:---:|------|:---:|
| A1 | Exploring Word Vectors | ✅ |
| A2 | Dependency Parsing | ⬜ |
| A3 | Neural Machine Translation | ⬜ |
| A4 | Self-Attention, Transformers, and Pretraining | 🚧 |

## 目录结构

```text
a1/
  student/                     # A1 assignment 代码
a2/
  notes/                       # A2 笔记
  student/                     # A2 assignment 代码
a3/
  note/                        # A3 笔记
  student/                     # A3 assignment 代码
a4/
  student/                     # A4 assignment 代码和数据
  transformer_from_scratch/    # 手写 Transformer 学习实现
notes/                         # 课程 lecture notes
```

## Transformer From Scratch

`a4/transformer_from_scratch/` 用来手写 Transformer 的核心结构，保留从展开版
学习代码到整理版模型的演进过程：

```text
main.py                    # 展开式学习过程，保留中间 tensor 观察和注释
decoder_block_clean.py     # Step 1-8 的整理版 decoder block
mini_transformer_lm.py     # Step 9: decoder-only MiniTransformerLM
mini_encoder_decoder.py    # Step 10: encoder-decoder + cross-attention
```

运行示例：

```bash
python3 a4/transformer_from_scratch/main.py
python3 a4/transformer_from_scratch/decoder_block_clean.py
python3 a4/transformer_from_scratch/mini_transformer_lm.py
python3 a4/transformer_from_scratch/mini_encoder_decoder.py
```
