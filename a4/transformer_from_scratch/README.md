# transformer_from_scratch

按笔记顺序手写一个最小 Transformer。你自己写核心代码，我只提供路线、骨架、检查点和调试建议。

## Step 1

目标：先让项目跑起来，只验证最小语言模型管线：

```text
token ids -> token embedding -> logits -> loss
```

运行：

```bash
../../.venv/bin/python main.py
```

当前固定规模：

```text
BATCH_SIZE = 2
SEQ_LEN = 4
VOCAB_SIZE = 16
D_MODEL = 8
```

你每填一个 TODO，就运行一次。
