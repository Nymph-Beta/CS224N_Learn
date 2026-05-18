from __future__ import annotations

"""
Step 10: build a tiny encoder-decoder Transformer.

这个文件补上原始 Transformer 里的 encoder、decoder 和 cross-attention。

和 decoder-only LM 的区别：

- Encoder self-attention 不加 causal mask，可以看完整 source sequence。
- Decoder self-attention 加 causal mask，只能看已经生成过的 target token。
- Cross-attention 的 Q 来自 decoder hidden states，K/V 来自 encoder memory。

当前整体结构：

    source token ids
    -> source embedding + source position embedding
    -> N 层 EncoderBlock
    -> encoder memory

    target input token ids
    -> target embedding + target position embedding
    -> N 层 DecoderBlock
       - masked self-attention
       - cross-attention over encoder memory
       - FFN
    -> final layer norm
    -> lm head
    -> logits

这里仍然手写多头注意力，不直接使用 nn.Transformer / nn.MultiheadAttention。
"""

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


BATCH_SIZE = 2
SRC_SEQ_LEN = 5
TGT_SEQ_LEN = 4
VOCAB_SIZE = 16
D_MODEL = 8
D_FF = 16
MAX_SEQ_LEN = 8
NUM_HEADS = 2
NUM_LAYERS = 2
HEAD_DIM = D_MODEL // NUM_HEADS

assert D_MODEL % NUM_HEADS == 0


def assert_shape(name: str, tensor: Tensor, expected_shape: tuple[int, ...]) -> None:
    """学习用检查工具：shape 不符合预期时立即报错。"""
    actual_shape = tuple(tensor.shape)
    if actual_shape != expected_shape:
        raise AssertionError(
            f"{name} shape mismatch: expected {expected_shape}, got {actual_shape}"
        )


def show_shape(name: str, tensor: Tensor) -> None:
    print(f"{name:>26}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")


def make_causal_mask(seq_len: int, device: torch.device) -> Tensor:
    # True 表示需要屏蔽的未来位置。
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )


def assert_attention_rows_sum_to_one(name: str, attention_weights: Tensor) -> None:
    # attention_weights: [B, H, query_len, key_len]。
    row_sums = attention_weights.sum(dim=-1)
    expected_row_sums = torch.ones_like(row_sums)
    if not torch.allclose(row_sums, expected_row_sums, atol=1e-6):
        max_diff = (row_sums - expected_row_sums).abs().max().item()
        raise AssertionError(
            f"{name} rows do not sum to 1: max difference is {max_diff:.4e}"
        )


def assert_no_attention_to_future(name: str, attention_weights: Tensor) -> None:
    # 只用于 decoder masked self-attention。
    seq_len = attention_weights.shape[-1]
    future_mask = make_causal_mask(seq_len, attention_weights.device)
    future_weights = attention_weights[:, :, future_mask]
    if not torch.allclose(future_weights, torch.zeros_like(future_weights), atol=1e-6):
        max_future_weight = future_weights.abs().max().item()
        raise AssertionError(
            f"{name} should not attend to future tokens, "
            f"max_future_weight={max_future_weight:.8f}"
        )


class MultiHeadAttention(nn.Module):
    """
    通用多头注意力模块。

    self-attention:
        query_states 和 key_value_states 是同一个张量。

    cross-attention:
        query_states 来自 decoder，key_value_states 来自 encoder memory。
    """

    def __init__(self) -> None:
        super().__init__()

        self.q_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.output_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(
        self,
        query_states: Tensor,
        key_value_states: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size, query_len, query_dim = query_states.shape
        kv_batch_size, key_len, kv_dim = key_value_states.shape

        assert batch_size == kv_batch_size
        assert query_dim == D_MODEL
        assert kv_dim == D_MODEL

        q = self.q_projection(query_states)
        k = self.k_projection(key_value_states)
        v = self.v_projection(key_value_states)

        q = q.reshape(batch_size, query_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.reshape(batch_size, key_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.reshape(batch_size, key_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)

        # scores: [B, H, query_len, key_len]。
        # cross-attention 时 query_len 可以不同于 key_len。
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)

        if attention_mask is not None:
            # attention_mask 通常是 [query_len, key_len]。
            # PyTorch 会把它 broadcast 到 [B, H, query_len, key_len]。
            scores = scores.masked_fill(attention_mask, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        assert_attention_rows_sum_to_one("attention", attention_weights)

        attention_output = torch.matmul(attention_weights, v)
        assert_shape(
            "attention_output before merge",
            attention_output,
            (batch_size, NUM_HEADS, query_len, HEAD_DIM),
        )

        merged_attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, query_len, D_MODEL
        )
        projected_attention_output = self.output_projection(merged_attention_output)

        return projected_attention_output, attention_weights


class FeedForward(nn.Module):
    """每个 token 位置独立运行的 FFN。"""

    def __init__(self) -> None:
        super().__init__()

        self.up_projection = nn.Linear(D_MODEL, D_FF)
        self.down_projection = nn.Linear(D_FF, D_MODEL)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down_projection(F.relu(self.up_projection(hidden)))


class EncoderBlock(nn.Module):
    """Encoder block：不加 causal mask，可以看完整 source sequence。"""

    def __init__(self) -> None:
        super().__init__()

        self.self_attention_layer_norm = nn.LayerNorm(D_MODEL)
        self.ffn_layer_norm = nn.LayerNorm(D_MODEL)
        self.self_attention = MultiHeadAttention()
        self.feed_forward = FeedForward()

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        attention_input = self.self_attention_layer_norm(hidden)
        attention_output, attention_weights = self.self_attention(
            query_states=attention_input,
            key_value_states=attention_input,
        )
        hidden = hidden + attention_output

        ffn_input = self.ffn_layer_norm(hidden)
        ffn_output = self.feed_forward(ffn_input)
        hidden = hidden + ffn_output

        return hidden, attention_weights


class DecoderBlock(nn.Module):
    """Decoder block：masked self-attention + cross-attention + FFN。"""

    def __init__(self) -> None:
        super().__init__()

        self.self_attention_layer_norm = nn.LayerNorm(D_MODEL)
        self.cross_attention_layer_norm = nn.LayerNorm(D_MODEL)
        self.ffn_layer_norm = nn.LayerNorm(D_MODEL)

        self.self_attention = MultiHeadAttention()
        self.cross_attention = MultiHeadAttention()
        self.feed_forward = FeedForward()

    def forward(
        self, hidden: Tensor, encoder_memory: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, target_len, _ = hidden.shape
        causal_mask = make_causal_mask(target_len, hidden.device)

        # 1. decoder masked self-attention：只能看 target 端过去和当前位置。
        self_attention_input = self.self_attention_layer_norm(hidden)
        self_attention_output, self_attention_weights = self.self_attention(
            query_states=self_attention_input,
            key_value_states=self_attention_input,
            attention_mask=causal_mask,
        )
        assert_no_attention_to_future("decoder self-attention", self_attention_weights)
        hidden = hidden + self_attention_output

        # 2. cross-attention：Q 来自 decoder，K/V 来自 encoder memory。
        # 这一步让 decoder 的每个位置读取 source sequence 的信息。
        cross_attention_input = self.cross_attention_layer_norm(hidden)
        cross_attention_output, cross_attention_weights = self.cross_attention(
            query_states=cross_attention_input,
            key_value_states=encoder_memory,
        )
        hidden = hidden + cross_attention_output

        # 3. FFN：对每个 target 位置独立做非线性变换。
        ffn_input = self.ffn_layer_norm(hidden)
        ffn_output = self.feed_forward(ffn_input)
        hidden = hidden + ffn_output

        assert_shape(
            "DecoderBlock hidden",
            hidden,
            (batch_size, target_len, D_MODEL),
        )
        return hidden, self_attention_weights, cross_attention_weights


class MiniEncoderDecoder(nn.Module):
    """Step 10 的最小 encoder-decoder Transformer。"""

    def __init__(self) -> None:
        super().__init__()

        self.source_token_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.target_token_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.source_position_embedding = nn.Embedding(MAX_SEQ_LEN, D_MODEL)
        self.target_position_embedding = nn.Embedding(MAX_SEQ_LEN, D_MODEL)

        self.encoder_blocks = nn.ModuleList([EncoderBlock() for _ in range(NUM_LAYERS)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock() for _ in range(NUM_LAYERS)])

        self.encoder_final_layer_norm = nn.LayerNorm(D_MODEL)
        self.decoder_final_layer_norm = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def encode(self, source_token_ids: Tensor) -> tuple[Tensor, list[Tensor]]:
        batch_size, source_len = source_token_ids.shape
        assert source_len <= MAX_SEQ_LEN

        source_position_ids = torch.arange(
            source_len,
            dtype=torch.long,
            device=source_token_ids.device,
        )
        hidden = self.source_token_embedding(source_token_ids)
        hidden = hidden + self.source_position_embedding(source_position_ids)
        assert_shape("encoder input hidden", hidden, (batch_size, source_len, D_MODEL))

        encoder_attention_history: list[Tensor] = []
        for block in self.encoder_blocks:
            hidden, attention_weights = block(hidden)
            encoder_attention_history.append(attention_weights)

        encoder_memory = self.encoder_final_layer_norm(hidden)
        assert_shape(
            "encoder_memory",
            encoder_memory,
            (batch_size, source_len, D_MODEL),
        )
        return encoder_memory, encoder_attention_history

    def decode(
        self, target_token_ids: Tensor, encoder_memory: Tensor
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        batch_size, target_len = target_token_ids.shape
        assert target_len <= MAX_SEQ_LEN

        target_position_ids = torch.arange(
            target_len,
            dtype=torch.long,
            device=target_token_ids.device,
        )
        hidden = self.target_token_embedding(target_token_ids)
        hidden = hidden + self.target_position_embedding(target_position_ids)
        assert_shape("decoder input hidden", hidden, (batch_size, target_len, D_MODEL))

        decoder_self_attention_history: list[Tensor] = []
        cross_attention_history: list[Tensor] = []
        for block in self.decoder_blocks:
            hidden, self_attention_weights, cross_attention_weights = block(
                hidden,
                encoder_memory,
            )
            decoder_self_attention_history.append(self_attention_weights)
            cross_attention_history.append(cross_attention_weights)

        hidden = self.decoder_final_layer_norm(hidden)
        logits = self.lm_head(hidden)
        assert_shape("logits", logits, (batch_size, target_len, VOCAB_SIZE))

        return logits, decoder_self_attention_history, cross_attention_history

    def forward(
        self, source_token_ids: Tensor, target_token_ids: Tensor
    ) -> tuple[Tensor, list[Tensor], list[Tensor], list[Tensor]]:
        encoder_memory, encoder_attention_history = self.encode(source_token_ids)
        logits, decoder_self_attention_history, cross_attention_history = self.decode(
            target_token_ids,
            encoder_memory,
        )

        return (
            logits,
            encoder_attention_history,
            decoder_self_attention_history,
            cross_attention_history,
        )


def make_toy_batch(device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    # source 是 encoder 读入的完整输入序列。
    source_token_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
        ],
        dtype=torch.long,
        device=device,
    )

    # target_input 是 teacher forcing 下喂给 decoder 的已知前缀。
    # 这里用 0 当作 toy BOS token。
    target_input_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [0, 6, 7, 8],
        ],
        dtype=torch.long,
        device=device,
    )

    # targets 是 decoder 每个位置应该预测的答案。
    targets = torch.tensor(
        [
            [1, 2, 3, 4],
            [6, 7, 8, 9],
        ],
        dtype=torch.long,
        device=device,
    )
    return source_token_ids, target_input_ids, targets


def compute_loss(logits: Tensor, targets: Tensor) -> Tensor:
    batch_size, target_len, vocab_size = logits.shape
    return F.cross_entropy(
        logits.reshape(batch_size * target_len, vocab_size),
        targets.reshape(batch_size * target_len),
    )


def train_on_toy_batch(
    model: MiniEncoderDecoder,
    source_token_ids: Tensor,
    target_input_ids: Tensor,
    targets: Tensor,
) -> None:
    # 这个训练循环只验证 encoder-decoder 的前向、反向和 cross-attention 参数都能参与优化。
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-2)

    with torch.no_grad():
        initial_loss = compute_loss(
            model(source_token_ids, target_input_ids)[0],
            targets,
        ).item()

    for _ in range(100):
        optimizer.zero_grad()
        logits = model(source_token_ids, target_input_ids)[0]
        loss = compute_loss(logits, targets)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = compute_loss(
            model(source_token_ids, target_input_ids)[0],
            targets,
        ).item()

    print(f"initial_loss: {initial_loss:.4f}")
    print(f"final_loss:   {final_loss:.4f}")
    if final_loss >= initial_loss:
        raise AssertionError("toy encoder-decoder training loss did not decrease")


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")

    print("Step 10: encoder-decoder MiniTransformer")
    print()

    model = MiniEncoderDecoder().to(device)
    source_token_ids, target_input_ids, targets = make_toy_batch(device)

    (
        logits,
        encoder_attention_history,
        decoder_self_attention_history,
        cross_attention_history,
    ) = model(source_token_ids, target_input_ids)

    show_shape("source_token_ids", source_token_ids)
    show_shape("target_input_ids", target_input_ids)
    show_shape("targets", targets)
    show_shape("logits", logits)
    show_shape("encoder layer0 attention", encoder_attention_history[0])
    show_shape("decoder layer0 self attention", decoder_self_attention_history[0])
    show_shape("decoder layer0 cross attention", cross_attention_history[0])
    print(f"loss before training: {compute_loss(logits, targets).item():.4f}")
    print()

    train_on_toy_batch(model, source_token_ids, target_input_ids, targets)
    print("ok: MiniEncoderDecoder is runnable and toy loss decreases")


if __name__ == "__main__":
    main()
