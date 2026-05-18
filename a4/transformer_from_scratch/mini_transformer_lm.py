from __future__ import annotations

"""
Step 9: build a decoder-only MiniTransformerLM.

这个文件和 main.py 的分工不同：

- main.py 保留逐步展开的学习过程，方便观察每个中间张量。
- mini_transformer_lm.py 把已经学过的模块整理成一个完整的 decoder-only LM。

当前模型结构：

    token ids
    -> token embedding + position embedding
    -> N 层 DecoderBlock
    -> final layer norm
    -> lm head
    -> logits

这里仍然不使用 nn.Transformer / nn.MultiheadAttention，而是手写 Q/K/V、
causal mask、多头拆分和 FFN，方便对照 main.py 里的学习痕迹。
"""

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# 为了学习阶段容易观察，继续使用很小的模型规模。
BATCH_SIZE = 2
SEQ_LEN = 4
VOCAB_SIZE = 16
D_MODEL = 8
D_FF = 16
MAX_SEQ_LEN = 8
NUM_HEADS = 2
NUM_LAYERS = 2
HEAD_DIM = D_MODEL // NUM_HEADS

# 多头注意力要求每个 head 均分 hidden 维度。
assert D_MODEL % NUM_HEADS == 0


def assert_shape(name: str, tensor: Tensor, expected_shape: tuple[int, ...]) -> None:
    """学习用检查工具：shape 不符合预期时尽早失败。"""
    actual_shape = tuple(tensor.shape)
    if actual_shape != expected_shape:
        raise AssertionError(
            f"{name} shape mismatch: expected {expected_shape}, got {actual_shape}"
        )


def show_shape(name: str, tensor: Tensor) -> None:
    """只打印 shape，避免完整训练循环输出太多数值。"""
    print(f"{name:>24}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")


def make_causal_mask(seq_len: int, device: torch.device) -> Tensor:
    # causal mask 的 True 位置表示“未来 token”，这些位置会被 masked_fill 成 -inf。
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )


def assert_attention_rows_sum_to_one(name: str, attention_weights: Tensor) -> None:
    # attention_weights: [B, H, L, L]。
    # softmax 在最后一维 key_position 上做，所以每个 query 行都应该加总为 1。
    row_sums = attention_weights.sum(dim=-1)
    expected_row_sums = torch.ones_like(row_sums)
    if not torch.allclose(row_sums, expected_row_sums, atol=1e-6):
        max_diff = (row_sums - expected_row_sums).abs().max().item()
        raise AssertionError(
            f"{name} rows do not sum to 1: max difference is {max_diff:.4e}"
        )


def assert_no_attention_to_future(name: str, attention_weights: Tensor) -> None:
    # decoder-only LM 必须保证当前位置不能 attend 到未来 token。
    seq_len = attention_weights.shape[-1]
    future_mask = make_causal_mask(seq_len, attention_weights.device)
    future_weights = attention_weights[:, :, future_mask]
    if not torch.allclose(future_weights, torch.zeros_like(future_weights), atol=1e-6):
        max_future_weight = future_weights.abs().max().item()
        raise AssertionError(
            f"{name} should not attend to future tokens, "
            f"max_future_weight={max_future_weight:.8f}"
        )


class MultiHeadCausalSelfAttention(nn.Module):
    """手写多头 causal self-attention。"""

    def __init__(self) -> None:
        super().__init__()

        # Q/K/V 都来自同一份 hidden states，所以这是 self-attention。
        self.q_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)

        # 多个 head concat 回 D_MODEL 后，再用输出投影混合各个 head 的信息。
        self.output_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, seq_len, d_model = hidden.shape
        assert d_model == D_MODEL

        q = self.q_projection(hidden)
        k = self.k_projection(hidden)
        v = self.v_projection(hidden)

        # [B, L, D] -> [B, L, H, HEAD_DIM] -> [B, H, L, HEAD_DIM]。
        # 每个 head 在自己的 HEAD_DIM 子空间里独立计算 attention。
        q = q.reshape(batch_size, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)

        # scores: [B, H, L, L]，每个 query 位置都会对所有 key 位置打分。
        # 缩放项使用 sqrt(HEAD_DIM)，因为每个 head 的点积只发生在 HEAD_DIM 维空间。
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)

        causal_mask = make_causal_mask(seq_len, scores.device)
        masked_scores = scores.masked_fill(causal_mask, float("-inf"))
        attention_weights = F.softmax(masked_scores, dim=-1)

        assert_attention_rows_sum_to_one("causal self-attention", attention_weights)
        assert_no_attention_to_future("causal self-attention", attention_weights)

        attention_output = torch.matmul(attention_weights, v)
        assert_shape(
            "attention_output before merge",
            attention_output,
            (batch_size, NUM_HEADS, seq_len, HEAD_DIM),
        )

        # [B, H, L, HEAD_DIM] -> [B, L, H, HEAD_DIM] -> [B, L, D]。
        merged_attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, seq_len, D_MODEL
        )

        projected_attention_output = self.output_projection(merged_attention_output)
        return projected_attention_output, attention_weights


class FeedForward(nn.Module):
    """Transformer block 里的 position-wise FFN。"""

    def __init__(self) -> None:
        super().__init__()

        # 先升维再降维：升维提供更大的中间表达空间，非线性激活后再回到主干维度。
        self.up_projection = nn.Linear(D_MODEL, D_FF)
        self.down_projection = nn.Linear(D_FF, D_MODEL)

    def forward(self, hidden: Tensor) -> Tensor:
        ffn_hidden = self.up_projection(hidden)
        ffn_activated = F.relu(ffn_hidden)
        return self.down_projection(ffn_activated)


class DecoderBlock(nn.Module):
    """一个 pre-norm decoder block。"""

    def __init__(self) -> None:
        super().__init__()

        self.attention_layer_norm = nn.LayerNorm(D_MODEL)
        self.ffn_layer_norm = nn.LayerNorm(D_MODEL)
        self.self_attention = MultiHeadCausalSelfAttention()
        self.feed_forward = FeedForward()

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        # pre-norm 写法：先 norm，再进子模块，最后 residual add。
        attention_input = self.attention_layer_norm(hidden)
        attention_output, attention_weights = self.self_attention(attention_input)
        hidden = hidden + attention_output

        ffn_input = self.ffn_layer_norm(hidden)
        ffn_output = self.feed_forward(ffn_input)
        hidden = hidden + ffn_output

        return hidden, attention_weights


class MiniTransformerLM(nn.Module):
    """Step 9 的 decoder-only MiniTransformerLM。"""

    def __init__(self) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.position_embedding = nn.Embedding(MAX_SEQ_LEN, D_MODEL)
        self.blocks = nn.ModuleList([DecoderBlock() for _ in range(NUM_LAYERS)])
        self.final_layer_norm = nn.LayerNorm(D_MODEL)

        # lm_head 把每个位置的 D_MODEL hidden vector 投影成词表 logits。
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(
        self, token_ids: Tensor, return_attention: bool = False
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        batch_size, seq_len = token_ids.shape
        assert seq_len <= MAX_SEQ_LEN

        position_ids = torch.arange(seq_len, dtype=torch.long, device=token_ids.device)

        # token embedding 表示“这个 token 是谁”，position embedding 表示“它在第几个位置”。
        hidden = self.token_embedding(token_ids) + self.position_embedding(position_ids)
        assert_shape("hidden after embedding", hidden, (batch_size, seq_len, D_MODEL))

        attention_history: list[Tensor] = []
        for block in self.blocks:
            hidden, attention_weights = block(hidden)
            attention_history.append(attention_weights)

        hidden = self.final_layer_norm(hidden)
        logits = self.lm_head(hidden)
        assert_shape("logits", logits, (batch_size, seq_len, VOCAB_SIZE))

        if return_attention:
            return logits, attention_history
        return logits


def make_toy_batch(device: torch.device) -> tuple[Tensor, Tensor]:
    # 玩具任务：每个位置预测下一个 token。
    # 输入  [1, 2, 3, 4]，目标 [2, 3, 4, 5]。
    token_ids = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ],
        dtype=torch.long,
        device=device,
    )
    targets = torch.tensor(
        [
            [2, 3, 4, 5],
            [6, 7, 8, 9],
        ],
        dtype=torch.long,
        device=device,
    )
    return token_ids, targets


def compute_loss(logits: Tensor, targets: Tensor) -> Tensor:
    batch_size, seq_len, vocab_size = logits.shape
    return F.cross_entropy(
        logits.reshape(batch_size * seq_len, vocab_size),
        targets.reshape(batch_size * seq_len),
    )


def train_on_toy_batch(model: MiniTransformerLM, token_ids: Tensor, targets: Tensor) -> None:
    # 这里只是验证训练链路能跑通，并且固定 toy batch 上 loss 可以下降。
    # 这不是正式语言模型训练，不需要追求泛化能力。
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-2)

    with torch.no_grad():
        initial_loss = compute_loss(model(token_ids), targets).item()

    for _ in range(80):
        optimizer.zero_grad()
        logits = model(token_ids)
        loss = compute_loss(logits, targets)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = compute_loss(model(token_ids), targets).item()

    print(f"initial_loss: {initial_loss:.4f}")
    print(f"final_loss:   {final_loss:.4f}")
    if final_loss >= initial_loss:
        raise AssertionError("toy training loss did not decrease")


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")

    print("Step 9: decoder-only MiniTransformerLM")
    print()

    model = MiniTransformerLM().to(device)
    token_ids, targets = make_toy_batch(device)

    logits, attention_history = model(token_ids, return_attention=True)

    show_shape("token_ids", token_ids)
    show_shape("targets", targets)
    show_shape("logits", logits)
    show_shape("layer0 attention", attention_history[0])
    print(f"loss before training: {compute_loss(logits, targets).item():.4f}")
    print()

    train_on_toy_batch(model, token_ids, targets)
    print("ok: MiniTransformerLM is runnable and toy loss decreases")


if __name__ == "__main__":
    main()
