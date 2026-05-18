from __future__ import annotations

"""
整理版 Step 1-8：把 main.py 里已经手写跑通的 decoder block 结构整理成清晰模块。

这个文件的定位：

- main.py 继续保留学习痕迹、展开计算、旧代码注释和大量 tensor 打印。
- decoder_block_clean.py 用更干净的 class 结构复盘同一条主线。
- mini_transformer_lm.py 再继续把 decoder block 堆叠成完整 decoder-only LM。

当前整理后的流程：

    token ids
    -> token embedding + position embedding
    -> MultiHeadCausalSelfAttention
    -> residual add
    -> FeedForward
    -> residual add
    -> output projection
    -> logits
    -> loss

这里仍然手写 Q/K/V、多头拆分、causal mask、softmax 和 AV，
不使用 nn.Transformer / nn.MultiheadAttention。
"""

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# 继续沿用 main.py 里的小尺寸，方便和学习版逐行对照。
BATCH_SIZE = 2
SEQ_LEN = 4
VOCAB_SIZE = 16
D_MODEL = 8
MAX_SEQ_LEN = 8
D_FF = 16
NUM_HEADS = 2
HEAD_DIM = D_MODEL // NUM_HEADS

# 多头注意力需要把 D_MODEL 均匀拆成 NUM_HEADS 份。
assert D_MODEL % NUM_HEADS == 0


def assert_shape(name: str, tensor: Tensor, expected_shape: tuple[int, ...]) -> None:
    """学习用 shape 检查：一旦形状不符合预期，就立刻报错。"""
    actual_shape = tuple(tensor.shape)
    if actual_shape != expected_shape:
        raise AssertionError(
            f"{name} shape mismatch: expected {expected_shape}, got {actual_shape}"
        )


def show_shape(name: str, tensor: Tensor) -> None:
    """整理版只打印关键 shape，不再打印每个 tensor 的完整数值。"""
    print(f"{name:>28}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")


def make_causal_mask(seq_len: int, device: torch.device) -> Tensor:
    # causal mask 是一个 [L, L] 布尔矩阵。
    # True 表示“未来位置”，后面会被 masked_fill 替换成 -inf。
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )


def assert_attention_rows_sum_to_one(name: str, attention_weights: Tensor) -> None:
    # attention_weights: [B, H, L, L]。
    # softmax 在最后一维 key_position 上做，所以每个 query 行的概率和应该是 1。
    row_sums = attention_weights.sum(dim=-1)
    expected_row_sums = torch.ones_like(row_sums)
    if not torch.allclose(row_sums, expected_row_sums, atol=1e-6):
        max_diff = (row_sums - expected_row_sums).abs().max().item()
        raise AssertionError(
            f"{name} rows do not sum to 1: max difference is {max_diff:.4e}"
        )


def assert_no_attention_to_future(name: str, attention_weights: Tensor) -> None:
    # causal self-attention 中，未来位置的 attention weight 应该是 0。
    seq_len = attention_weights.shape[-1]
    future_mask = make_causal_mask(seq_len, attention_weights.device)
    future_weights = attention_weights[:, :, future_mask]
    expected_future_weights = torch.zeros_like(future_weights)
    if not torch.allclose(future_weights, expected_future_weights, atol=1e-6):
        max_future_weight = future_weights.abs().max().item()
        raise AssertionError(
            f"{name} should not attend to future tokens, "
            f"max_future_weight={max_future_weight:.8f}"
        )


class TokenAndPositionEmbedding(nn.Module):
    """把 token id 转成 hidden states，并加入位置信息。"""

    def __init__(self) -> None:
        super().__init__()

        # token embedding 负责表示“这个 token 是谁”。
        self.token_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)

        # position embedding 负责表示“这个 token 在序列里的第几个位置”。
        self.position_embedding = nn.Embedding(MAX_SEQ_LEN, D_MODEL)

    def forward(self, token_ids: Tensor) -> Tensor:
        batch_size, seq_len = token_ids.shape
        if seq_len > MAX_SEQ_LEN:
            raise AssertionError("seq_len must be less than or equal to MAX_SEQ_LEN")

        position_ids = torch.arange(
            seq_len,
            dtype=torch.long,
            device=token_ids.device,
        )

        token_hidden = self.token_embedding(token_ids)
        position_hidden = self.position_embedding(position_ids)

        # position_hidden: [L, D] 会 broadcast 到 [B, L, D]。
        hidden = token_hidden + position_hidden
        assert_shape("embedded hidden", hidden, (batch_size, seq_len, D_MODEL))
        return hidden


class MultiHeadCausalSelfAttention(nn.Module):
    """整理版多头 causal self-attention。"""

    def __init__(self) -> None:
        super().__init__()

        # self-attention 的 Q/K/V 都来自同一份 hidden states。
        self.q_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)

        # 多头结果 concat 回 D_MODEL 后，再经过输出投影混合各个 head 的信息。
        self.output_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, seq_len, d_model = hidden.shape
        assert d_model == D_MODEL

        q = self.q_projection(hidden)
        k = self.k_projection(hidden)
        v = self.v_projection(hidden)

        # [B, L, D] -> [B, L, H, HEAD_DIM] -> [B, H, L, HEAD_DIM]。
        # 这样每个 head 都可以独立计算自己的注意力矩阵。
        q = q.reshape(batch_size, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)

        assert_shape("q after split", q, (batch_size, NUM_HEADS, seq_len, HEAD_DIM))
        assert_shape("k after split", k, (batch_size, NUM_HEADS, seq_len, HEAD_DIM))
        assert_shape("v after split", v, (batch_size, NUM_HEADS, seq_len, HEAD_DIM))

        # scores: [B, H, L, L]。
        # 每个 head 的每个 query 位置，都会对所有 key 位置做一次点积打分。
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)

        causal_mask = make_causal_mask(seq_len, scores.device)
        masked_scores = scores.masked_fill(causal_mask, float("-inf"))
        attention_weights = F.softmax(masked_scores, dim=-1)

        assert_attention_rows_sum_to_one("attention_weights", attention_weights)
        assert_no_attention_to_future("attention_weights", attention_weights)

        # 这里就是注意力流程里的 AV：
        # attention_weights: [B, H, L, L]
        # v:                 [B, H, L, HEAD_DIM]
        # attention_output:  [B, H, L, HEAD_DIM]
        attention_output = torch.matmul(attention_weights, v)

        # [B, H, L, HEAD_DIM] -> [B, L, H, HEAD_DIM] -> [B, L, D]。
        merged_attention_output = attention_output.transpose(1, 2).reshape(
            batch_size,
            seq_len,
            D_MODEL,
        )
        projected_attention_output = self.output_projection(merged_attention_output)

        return projected_attention_output, attention_weights


class FeedForward(nn.Module):
    """Transformer block 里的 position-wise FFN。"""

    def __init__(self) -> None:
        super().__init__()

        # FFN 先升维，经过非线性激活，再降回 D_MODEL。
        self.up_projection = nn.Linear(D_MODEL, D_FF)
        self.down_projection = nn.Linear(D_FF, D_MODEL)

    def forward(self, hidden: Tensor) -> Tensor:
        ffn_hidden = self.up_projection(hidden)
        ffn_activated = F.relu(ffn_hidden)
        return self.down_projection(ffn_activated)


class DecoderBlock(nn.Module):
    """一个整理后的 pre-norm decoder block。"""

    def __init__(self) -> None:
        super().__init__()

        self.attention_layer_norm = nn.LayerNorm(D_MODEL)
        self.ffn_layer_norm = nn.LayerNorm(D_MODEL)
        self.self_attention = MultiHeadCausalSelfAttention()
        self.feed_forward = FeedForward()

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        # 第一段：LayerNorm -> masked multi-head self-attention -> residual add。
        attention_input = self.attention_layer_norm(hidden)
        attention_output, attention_weights = self.self_attention(attention_input)
        hidden_after_attention = hidden + attention_output

        # 第二段：LayerNorm -> FFN -> residual add。
        ffn_input = self.ffn_layer_norm(hidden_after_attention)
        ffn_output = self.feed_forward(ffn_input)
        hidden_after_ffn = hidden_after_attention + ffn_output

        return hidden_after_ffn, attention_weights


class CleanDecoderBlockDemo(nn.Module):
    """用一个 decoder block 跑通 token ids -> logits 的整理版 demo。"""

    def __init__(self) -> None:
        super().__init__()

        self.embedding = TokenAndPositionEmbedding()
        self.decoder_block = DecoderBlock()
        self.output_projection = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.embedding(token_ids)
        block_output, attention_weights = self.decoder_block(hidden)
        logits = self.output_projection(block_output)

        batch_size, seq_len = token_ids.shape
        assert_shape("logits", logits, (batch_size, seq_len, VOCAB_SIZE))
        return logits, attention_weights


def compute_loss(logits: Tensor, targets: Tensor) -> Tensor:
    batch_size, seq_len, vocab_size = logits.shape
    return F.cross_entropy(
        logits.reshape(batch_size * seq_len, vocab_size),
        targets.reshape(batch_size * seq_len),
    )


def main() -> None:
    torch.manual_seed(0)

    print("Clean Step 1-8: token ids -> decoder block -> logits -> loss")
    print()

    token_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ],
        dtype=torch.long,
    )

    assert_shape("token_ids", token_ids, (BATCH_SIZE, SEQ_LEN))
    assert_shape("targets", targets, (BATCH_SIZE, SEQ_LEN))

    model = CleanDecoderBlockDemo()
    logits, attention_weights = model(token_ids)
    loss = compute_loss(logits, targets)

    show_shape("token_ids", token_ids)
    show_shape("targets", targets)
    show_shape("logits", logits)
    show_shape("attention_weights", attention_weights)

    # 只展示第一个样本、第一个 head 的 attention matrix。
    # 这样可以看 causal mask 是否形成下三角结构，但不会像 main.py 那样输出大量张量。
    print()
    print("attention_weights[0, 0]: query rows x key columns")
    print(attention_weights[0, 0])
    print()
    print(f"loss: {loss.item():.4f}")
    print("ok: clean decoder block demo is runnable")


if __name__ == "__main__":
    main()
