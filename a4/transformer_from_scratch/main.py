from __future__ import annotations

"""
Step 1: build the smallest runnable language-model pipeline.

Learning order:

    1. Create token_ids first.
    2. Send token_ids through an embedding table.
    3. Send hidden states through an output projection.
    4. Compute a loss.
    5. Only after the above works, wrap the logic in TinyTokenModel.

Run after every TODO:

    ../.venv/bin/python main.py

The first part keeps the learning path explicit in main().
The second part packages the same path into TinyTokenModel.
"""

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math


# Fixed scale: keep the world tiny while learning.
BATCH_SIZE = 2
SEQ_LEN = 4
VOCAB_SIZE = 16
D_MODEL = 8
# 位置向量表允许的最大序列长度。
# 当前 SEQ_LEN=4，只会用到位置 0, 1, 2, 3；MAX_SEQ_LEN=8 表示表里预留了 8 个位置。
MAX_SEQ_LEN = 8

# FFN 中间层维度。Transformer 的 FFN 通常会先把 d_model 升到更大的维度，
# 做一次非线性变换后再降回 d_model。这里用 16 是为了让形状变化容易观察。
D_FF = 16

# 多头注意力的头数。这里保持很小，方便观察 shape。
NUM_HEADS = 2
# 每个 head 分到的维度。D_MODEL 会被拆成 NUM_HEADS 份：
# D_MODEL = NUM_HEADS * HEAD_DIM。
HEAD_DIM = D_MODEL // NUM_HEADS

# 多头注意力要求 D_MODEL 可以被 head 数整除，否则无法均匀切分每个 head。
assert D_MODEL % NUM_HEADS == 0


# These are only semantic names for Tensor roles.
# Python/PyTorch will not check these shapes automatically.
TokenIds = Tensor  # shape: [BATCH_SIZE, SEQ_LEN], dtype: torch.long
HiddenStates = Tensor  # shape: [BATCH_SIZE, SEQ_LEN, D_MODEL]
Logits = Tensor  # shape: [BATCH_SIZE, SEQ_LEN, VOCAB_SIZE]

QueryStates = Tensor  # shape: [BATCH_SIZE, SEQ_LEN, D_MODEL]
KeyStates = Tensor  # shape: [BATCH_SIZE, SEQ_LEN, D_MODEL]
ValueStates = Tensor  # shape: [BATCH_SIZE, SEQ_LEN, D_MODEL]
AttentionScores = Tensor  # shape: [BATCH_SIZE, SEQ_LEN, SEQ_LEN]
AttentionWeights = Tensor  # shape: [BATCH_SIZE, SEQ_LEN, SEQ_LEN]
AttentionOutput = Tensor  # shape: [BATCH_SIZE, SEQ_LEN, D_MODEL]

# 这里的位置 id 不需要按 batch 复制一份，因为同一个 batch 内每条样本的位置编号相同。
PositionIds = Tensor  # shape: [SEQ_LEN], dtype: torch.long
# position_embedding 查表后得到每个位置的向量，之后会 broadcast 到 batch 维度。
PositionStates = Tensor  # shape: [SEQ_LEN, D_MODEL]

def assert_shape(name: str, tensor: Tensor, expected_shape: tuple[int, ...]) -> None:
    """Observation tool: fail fast when a tensor shape is not what you expect."""
    actual_shape = tuple(tensor.shape)
    if actual_shape != expected_shape:
        raise AssertionError(
            f"{name} shape mismatch: expected {expected_shape}, got {actual_shape}"
        )


def show_tensor(name: str, tensor: Tensor) -> None:
    """Debug representation. Use it to inspect learning-time values and shapes."""
    print(f"{name:>12}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    print(tensor)
    print()


def stop_at(todo_name: str, goal: str) -> None:
    print(f"{todo_name} is not filled yet.")
    print(goal)
    print("After filling it, run this file again.")

def assert_attention_rows_sum_to_one(name: str, attention_weights: AttentionWeights) -> None:
    # 这是一个学习阶段的检查点：如果 softmax 维度选对了，
    # attention_weights 的每一行相加都应该是 1。
    row_sums = attention_weights.sum(dim=-1)

    # 检查 row_sums 的形状是否正确。它应该是 [BATCH_SIZE, SEQ_LEN]，每个位置的值是对应行的和。
    assert_shape("attention_weights row sums", row_sums, (BATCH_SIZE, SEQ_LEN))
    
    show_tensor("attention_weights row sums", row_sums)

    # 构造一个和 row_sums 形状完全相同的“标准答案”张量。
    # 因为每一行 attention 分布都应该加总为 1，所以这里每个位置的期望值都是 1。
    expected_row_nums = torch.ones_like(row_sums)

    # 浮点计算会有微小误差，所以不能用 == 逐元素硬比较。
    # torch.allclose 会允许很小的数值偏差，更适合检查 softmax 这种浮点结果。
    if not torch.allclose(row_sums, expected_row_nums, atol=1e-6):
        # 如果检查失败，找出实际行和与期望值 1 之间最大的绝对差。
        # 这个数能帮助判断问题有多严重：是正常浮点误差，还是 softmax 维度真的选错了。
        max_diff = (row_sums - expected_row_nums).abs().max().item()
        raise AssertionError(
            f"attention_weights rows do not sum to 1: max difference is {max_diff:.4e}"
        )
    
def show_attention_matrix(name: str, attention_weights: AttentionWeights) -> None:
    first_matrix = attention_weights[0]

    assert_shape(
        f"{name}[0]",
        first_matrix,
        (SEQ_LEN, SEQ_LEN),
    )

    print(f"{name}[0]: query rows x key columns")
    print(first_matrix)
    print()

def assert_no_attention_to_future(name: str, attention_weights: AttentionWeights) -> None:
    # 这是另一个学习阶段的检查点：被 causal mask 屏蔽掉的未来位置，
    # 在 softmax 之后的 attention weight 应该是 0。
    # 这里重新构造同样的上三角 mask，用来定位“未来 token”所在的列。
    future_mask = torch.triu(
            torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool, device=attention_weights.device),
            diagonal=1,
        )

    # attention_weights 的形状是 [batch, query_position, key_position]。
    # future_mask 的形状是 [query_position, key_position]。
    # attention_weights[:, future_mask] 会对每个 batch 取出所有被 mask 的未来位置权重，
    # 得到形状 [BATCH_SIZE, future_position_count] 的检查张量。
    future_weights = attention_weights[:, future_mask]

    # 如果 causal mask 生效，所有未来位置在 softmax 后都应该非常接近 0。
    # 这里用 allclose 而不是 ==，是为了允许浮点计算中的极小误差。
    if not torch.allclose(future_weights, torch.zeros_like(future_weights), atol=1e-6):
        max_future_weight = future_weights.abs().max().item()
        raise AssertionError(
            f"{name} should not attend to future tokens, max_future_weight={max_future_weight:.8f}"
        )


def assert_multihead_attention_rows_sum_to_one(
    name: str, attention_weights: Tensor
) -> None:
    # 多头版本的 attention_weights 形状是 [BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN]。
    # softmax 仍然在最后一维 key_position 上做，所以每个 batch、每个 head、每个 query 行都应该加总为 1。
    row_sums = attention_weights.sum(dim=-1)
    assert_shape(name + " row sums", row_sums, (BATCH_SIZE, NUM_HEADS, SEQ_LEN))
    show_tensor(name + " row sums", row_sums)

    expected_row_sums = torch.ones_like(row_sums)
    if not torch.allclose(row_sums, expected_row_sums, atol=1e-6):
        max_diff = (row_sums - expected_row_sums).abs().max().item()
        raise AssertionError(
            f"{name} rows do not sum to 1: max difference is {max_diff:.4e}"
        )


def assert_multihead_no_attention_to_future(name: str, attention_weights: Tensor) -> None:
    # 多头版本同样要验证 causal mask：每个 head 都不能看未来 token。
    # future_mask 仍是 [SEQ_LEN, SEQ_LEN]，会用于索引每个 head 的 query/key 矩阵。
    future_mask = torch.triu(
        torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool, device=attention_weights.device),
        diagonal=1,
    )

    # attention_weights[:, :, future_mask] 会保留 batch 和 head 两维，
    # 并从最后两维中取出所有未来位置的权重。
    # 结果形状是 [BATCH_SIZE, NUM_HEADS, future_position_count]。
    future_weights = attention_weights[:, :, future_mask]
    expected_future_weights = torch.zeros_like(future_weights)

    if not torch.allclose(future_weights, expected_future_weights, atol=1e-6):
        max_future_weight = future_weights.abs().max().item()
        raise AssertionError(
            f"{name} should not attend to future tokens, max_future_weight={max_future_weight:.8f}"
        )


class SingleHeadDecoderBlock(nn.Module):
    """
    Package the hand-written Step 6 flow into one reusable decoder block.

    It is the same structure you expanded in main():

        hidden
        -> LayerNorm
        -> causal single-head self-attention
        -> residual add
        -> LayerNorm
        -> FFN
        -> residual add
        -> block output
    """

    def __init__(self) -> None:
        super().__init__()

        # 两个 LayerNorm 分别服务于 attention 分支和 FFN 分支。
        # 这是 pre-norm 写法：每个子模块先 norm，再计算，再 residual add。
        self.attention_layer_norm = nn.LayerNorm(D_MODEL)
        self.ffn_layer_norm = nn.LayerNorm(D_MODEL)

        # 单头 self-attention 的 Q/K/V 投影。
        self.q_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)

        # position-wise FFN：D_MODEL -> D_FF -> D_MODEL。
        self.ffn_up_projection = nn.Linear(D_MODEL, D_FF)
        self.ffn_down_projection = nn.Linear(D_FF, D_MODEL)

    def forward(self, hidden: HiddenStates) -> HiddenStates:
        assert_shape("SingleHeadDecoderBlock input hidden", hidden, (BATCH_SIZE, SEQ_LEN, D_MODEL))

        # 先复用 main() 里已经跑通的 pre-norm attention 路径。
        attention_input = self.attention_layer_norm(hidden)

        q = self.q_projection(attention_input)
        k = self.k_projection(attention_input)
        v = self.v_projection(attention_input)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D_MODEL)
        causal_mask = torch.triu(
            torch.ones(SEQ_LEN, SEQ_LEN, device=scores.device, dtype=torch.bool),
            diagonal=1
        )
        masked_scores = scores.masked_fill(causal_mask, float("-inf"))
        attention_weights = F.softmax(masked_scores, dim=-1)

        assert_attention_rows_sum_to_one("SingleHeadDecoderBlock attention_weights", attention_weights)
        assert_no_attention_to_future("SingleHeadDecoderBlock attention_weights", attention_weights)

        attention_output = torch.matmul(attention_weights, v)
        hidden_after_attention = hidden + attention_output

        # 再复用 main() 里已经跑通的 pre-norm FFN 路径。
        ffn_input = self.ffn_layer_norm(hidden_after_attention)
        ffn_hidden = self.ffn_up_projection(ffn_input)
        ffn_activated = F.relu(ffn_hidden)
        ffn_output = self.ffn_down_projection(ffn_activated)
        hidden_after_ffn = hidden_after_attention + ffn_output

        assert_shape("SingleHeadDecoderBlock output hidden", hidden_after_ffn, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        return hidden_after_ffn

class MultiHeadDecoderBlock(nn.Module):
    """
    Package the same decoder block as SingleHeadDecoderBlock, but split Q/K/V
    into multiple heads before attention and concatenate them afterward.
    """

    def __init__(self) -> None:
        super().__init__()

        self.attention_layer_norm = nn.LayerNorm(D_MODEL)
        self.ffn_layer_norm = nn.LayerNorm(D_MODEL)

        self.q_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)

        self.output_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)

        self.ffn_up_projection = nn.Linear(D_MODEL, D_FF)
        self.ffn_down_projection = nn.Linear(D_FF, D_MODEL)
    
    def forward(self, hidden: HiddenStates) -> HiddenStates:
        assert_shape("MultiHeadDecoderBlock input hidden", hidden, (BATCH_SIZE, SEQ_LEN, D_MODEL))

        attention_input = self.attention_layer_norm(hidden)
        assert_shape("multihead attention_input", attention_input, (BATCH_SIZE, SEQ_LEN, D_MODEL))

        q = self.q_projection(attention_input)
        k = self.k_projection(attention_input)
        v = self.v_projection(attention_input)

        assert_shape("q before split", q, (BATCH_SIZE, SEQ_LEN, D_MODEL))

        # 把最后一维 D_MODEL 拆成 NUM_HEADS 个 head，每个 head 的维度是 HEAD_DIM。
        # reshape 后是 [BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM]。
        # transpose(1, 2) 把 head 维提前，得到注意力计算更方便的
        # [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]。
        q = q.reshape(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.reshape(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.reshape(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM).transpose(1, 2)

        assert_shape("q after split", q, (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        assert_shape("k after split", k, (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM))
        assert_shape("v after split", v, (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM))

        # 每个 head 独立计算自己的 scaled dot-product attention。
        # 这里除以 sqrt(HEAD_DIM)，不是 sqrt(D_MODEL)，因为每个 head 只在 HEAD_DIM 维空间里做点积。
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        assert_shape(
            "multihead scores",
            scores,
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN),
        )

        causal_mask = torch.triu(
            torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool, device=scores.device),
            diagonal=1,
        )

        masked_scores = scores.masked_fill(causal_mask, float("-inf"))
        attention_weights = F.softmax(masked_scores, dim=-1)

        assert_multihead_attention_rows_sum_to_one("multihead attention_weights", attention_weights)
        assert_multihead_no_attention_to_future("multihead attention_weights", attention_weights)

        attention_output = torch.matmul(attention_weights, v)

        assert_shape(
            "multihead attention_output",
            attention_output,
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM),
        )

        # 把多个 head 拼回主干维度。
        # 先 transpose 回 [BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM]，
        # 再 reshape 成 [BATCH_SIZE, SEQ_LEN, D_MODEL]。
        merged_attention_output = attention_output.transpose(1, 2).reshape(
            BATCH_SIZE, SEQ_LEN, D_MODEL
        )
        assert_shape("merged_attention_output", merged_attention_output, (BATCH_SIZE, SEQ_LEN, D_MODEL))

        # 多头 concat 之后通常还会接一个输出投影，把各 head 混合回统一表示空间。
        attention_output = self.output_projection(merged_attention_output)
        assert_shape("multihead projected attention_output", attention_output, (BATCH_SIZE, SEQ_LEN, D_MODEL))

        hidden_after_attention = hidden + attention_output

        ffn_input = self.ffn_layer_norm(hidden_after_attention)
        ffn_hidden = self.ffn_up_projection(ffn_input)
        ffn_activated = F.relu(ffn_hidden)
        ffn_output = self.ffn_down_projection(ffn_activated)

        hidden_after_ffn = hidden_after_attention + ffn_output
        assert_shape("MultiHeadDecoderBlock output hidden", hidden_after_ffn, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        return hidden_after_ffn


def main() -> None:
    torch.manual_seed(0)

    print("Step 1: token_ids -> embedding -> hidden -> projection -> logits -> loss")
    print()

    # TODO 1:
    # Create token_ids by hand.
    #
    # What you are making:
    # - A batch of 2 short token sequences.
    # - Each sequence has 4 token ids.
    # - Each id must be an integer in [0, VOCAB_SIZE).
    #
    # Expected shape:
    #   [BATCH_SIZE, SEQ_LEN] == [2, 4]
    #
    # Expected dtype:
    #   torch.long
    token_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ],
        dtype=torch.long,
    )

    assert_shape("token_ids", token_ids, (BATCH_SIZE, SEQ_LEN))
    show_tensor("token_ids", token_ids)

    # TODO 2:
    # Create an embedding table.
    #
    # Meaning:
    # - It has VOCAB_SIZE rows.
    # - Each row is a D_MODEL-dimensional vector.
    # - token_ids will be used to look up rows in this table.
    #
    # Input:
    #   token_ids, shape [BATCH_SIZE, SEQ_LEN]
    #
    # Output after lookup:
    #   hidden, shape [BATCH_SIZE, SEQ_LEN, D_MODEL]
    token_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)

    # 位置 embedding 表：第 i 行表示“第 i 个位置”的可学习向量。
    # token embedding 只知道“这个词是谁”，position embedding 补充“这个词在序列里的第几个位置”。
    position_embedding = nn.Embedding(MAX_SEQ_LEN, D_MODEL)

    # 位置 id 最大会到 SEQ_LEN - 1，所以位置表的行数必须至少覆盖当前序列长度。
    # 如果以后把 SEQ_LEN 调大但忘了调 MAX_SEQ_LEN，这里会尽早报错。
    assert SEQ_LEN <= MAX_SEQ_LEN, "SEQ_LEN must be less than or equal to MAX_SEQ_LEN for position embedding to work."

    # token_hidden 是纯 token id 查表结果：同一个 token id 在不同位置会得到同一个 token 向量。
    token_hidden = token_embedding(token_ids)

    # 生成当前序列的位置编号：[0, 1, 2, 3]。
    # device=token_ids.device 保证 position_ids 和 token_ids 在同一个设备上，
    # 之后如果把模型搬到 GPU，这里也不会出现 CPU/GPU 混用错误。
    position_ids = torch.arange(
        SEQ_LEN, dtype=torch.long,
        device=token_ids.device
    )

    # 根据位置编号查表，得到每个绝对位置对应的可学习位置向量。
    position_hidden = position_embedding(position_ids)

    # TODO 3:
    # Use token_embedding to convert token_ids into hidden states.
    # 原始版本只使用 token embedding：
    # hidden = token_embedding(token_ids)
    # 现在把 token 信息和位置信息相加，得到真正送入 attention 的输入表示。
    # token_hidden 的形状是 [BATCH_SIZE, SEQ_LEN, D_MODEL]，
    # position_hidden 的形状是 [SEQ_LEN, D_MODEL]；PyTorch 会把 position_hidden
    # 自动 broadcast 到 batch 维度，相当于每条样本都加上同一套位置向量。
    hidden = token_hidden + position_hidden

    # 这些检查用来确认每一步的张量形状符合预期，尤其是 position_hidden 的 broadcast 前形状。
    assert_shape("token_hidden", token_hidden, (BATCH_SIZE, SEQ_LEN, D_MODEL))
    assert_shape("position_ids", position_ids, (SEQ_LEN,))
    assert_shape("position_hidden", position_hidden, (SEQ_LEN, D_MODEL))
    assert_shape("hidden", hidden, (BATCH_SIZE, SEQ_LEN, D_MODEL))

    show_tensor("position_ids", position_ids)
    show_tensor("position_hidden", position_hidden)
    show_tensor("hidden", hidden)


    # 自注意力的第一步是把同一份 hidden states 映射成三种不同角色：
    # - q/query：当前位置主动发出的“查询”，用来问自己该关注哪些位置。
    # - k/key：每个位置提供的“索引标签”，用来和 query 计算匹配程度。
    # - v/value：每个位置真正携带的信息内容，最后会被加权汇总。
    # 这里先用单头注意力，所以 q/k/v 的最后一维仍保持 D_MODEL 不变。
    q_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)
    k_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)
    v_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)

    # Step 6: 加入 Add & Norm。
    # 这里采用 pre-norm 写法：先 LayerNorm，再进入 attention / FFN。
    # LayerNorm 不改变形状，只会在最后一维 D_MODEL 上做归一化，让每个 token 的表示更稳定。
    attention_layer_norm = nn.LayerNorm(D_MODEL)
    ffn_layer_norm = nn.LayerNorm(D_MODEL)

    # attention 分支的输入不再直接使用 hidden，而是先对 hidden 做 LayerNorm。
    # residual add 时仍然会把 attention_output 加回原始 hidden。
    attention_input = attention_layer_norm(hidden)
    assert_shape("attention_input", attention_input, (BATCH_SIZE, SEQ_LEN, D_MODEL))
    show_tensor("attention_input", attention_input)

    # 对每个 token 的 hidden vector 分别做线性变换。
    # 输入 attention_input 的形状是 [batch, seq_len, d_model]；
    # Linear 会作用在最后一维，因此 q/k/v 的形状仍是 [batch, seq_len, d_model]。
    # 原来的版本直接从 hidden 生成 q/k/v：
    # q = q_projection(hidden)
    # k = k_projection(hidden)
    # v = v_projection(hidden)
    # pre-norm 版本改成从 attention_input 生成 q/k/v。
    q = q_projection(attention_input)
    k = k_projection(attention_input)
    v = v_projection(attention_input)

    assert_shape("q", q, (BATCH_SIZE, SEQ_LEN, D_MODEL))
    assert_shape("k", k, (BATCH_SIZE, SEQ_LEN, D_MODEL))
    assert_shape("v", v, (BATCH_SIZE, SEQ_LEN, D_MODEL))

    # 计算每个 query 对所有 key 的相似度。
    # k.transpose(-2, -1) 把 key 从 [batch, seq_len, d_model]
    # 转成 [batch, d_model, seq_len]，这样 q @ k^T 的结果就是
    # [batch, seq_len, seq_len]：每一行表示某个当前位置对所有上下文位置的打分。
    # 除以 sqrt(D_MODEL) 是 scaled dot-product attention 的缩放项，
    # 用来避免 d_model 较大时点积数值过大，导致 softmax 过早饱和。
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D_MODEL)
    assert_shape("scores", scores, (BATCH_SIZE, SEQ_LEN, SEQ_LEN))
    show_tensor("scores", scores)

    # causal mask 用来禁止当前位置看见未来位置。
    # 先创建一个 [SEQ_LEN, SEQ_LEN] 的布尔矩阵，再用 torch.triu 只保留主对角线上方的部分。
    # diagonal=1 表示从主对角线右上方一格开始标 True：
    # - False 的位置保留，表示 query 可以 attend 到这些 key。
    # - True 的位置会被屏蔽，表示 query 不允许 attend 到未来 key。
    causal_mask = torch.triu(
        torch.ones(SEQ_LEN, SEQ_LEN, device=scores.device, dtype=torch.bool),
        diagonal=1
    )

    assert_shape("causal_mask", causal_mask, (SEQ_LEN, SEQ_LEN))
    show_tensor("causal_mask", causal_mask)

    # 把 causal_mask=True 的未来位置分数替换成 -inf。
    # 这里不是矩阵乘法，而是逐位置替换：
    # - mask=False 的位置保留原 scores。
    # - mask=True 的位置改成 -inf。
    # 因为 scores 是 [BATCH_SIZE, SEQ_LEN, SEQ_LEN]，causal_mask 是 [SEQ_LEN, SEQ_LEN]，
    # PyTorch 会把同一个 causal_mask broadcast 到 batch 里的每个样本。
    masked_scores = scores.masked_fill(causal_mask, float("-inf"))
    assert_shape("masked_scores", masked_scores, (BATCH_SIZE, SEQ_LEN, SEQ_LEN))
    show_tensor("masked_scores", masked_scores)


    # softmax 在最后一维上做归一化，也就是让每个 query 对所有 key 的分数
    # 变成一行概率分布。归一化后，每一行的和应该接近 1。
    # 原始版本直接对 scores 做 softmax，会允许每个位置看见未来 token：
    # attention_weights = F.softmax(scores, dim=-1)
    # 现在改成对 masked_scores 做 softmax；-inf 经过 softmax 后会变成 0，
    # 所以未来 token 的 attention weight 会被压到 0。
    attention_weights = F.softmax(masked_scores, dim=-1)

    assert_shape("attention_weights", attention_weights, (BATCH_SIZE, SEQ_LEN, SEQ_LEN))
    # 原来直接打印整个 batch 的 attention tensor，适合看完整数值：
    # show_tensor("attention_weights", attention_weights)
    # 现在改成只展示第一个样本的注意力矩阵，更容易观察 query 行和 key 列的关系。
    show_attention_matrix("attention_weights", attention_weights)

    # # 这是一个学习阶段的检查点：如果 softmax 维度选对了，
    # # attention_weights 的每一行相加都应该是 1。
    # row_sums = attention_weights.sum(dim=-1)

    # # 检查 row_sums 的形状是否正确。它应该是 [BATCH_SIZE, SEQ_LEN]，每个位置的值是对应行的和。
    # assert_shape("attention_weights row sums", row_sums, (BATCH_SIZE, SEQ_LEN))
    
    # show_tensor("attention_weights row sums", row_sums)

    # # 构造一个和 row_sums 形状完全相同的“标准答案”张量。
    # # 因为每一行 attention 分布都应该加总为 1，所以这里每个位置的期望值都是 1。
    # expected_row_nums = torch.ones_like(row_sums)

    # # 浮点计算会有微小误差，所以不能用 == 逐元素硬比较。
    # # torch.allclose 会允许很小的数值偏差，更适合检查 softmax 这种浮点结果。
    # if not torch.allclose(row_sums, expected_row_nums, atol=1e-6):
    #     # 如果检查失败，找出实际行和与期望值 1 之间最大的绝对差。
    #     # 这个数能帮助判断问题有多严重：是正常浮点误差，还是 softmax 维度真的选错了。
    #     max_diff = (row_sums - expected_row_nums).abs().max().item()
    #     raise AssertionError(
    #         f"attention_weights rows do not sum to 1: max difference is {max_diff:.4e}"
    #     )
    assert_attention_rows_sum_to_one("attention_weights", attention_weights)
    # assert_attention_rows_sum_to_one() 里面已经计算并打印过 row_sums。
    # 这里保留原来的重复打印代码作为学习记录，但先注释掉，避免运行时输出两遍。
    # row_sums = attention_weights.sum(dim=-1)
    # show_tensor("attention_weights row sums", row_sums)

    assert_no_attention_to_future("attention_weights", attention_weights)

    # 用注意力权重对 value 做加权求和。
    # attention_weights: [batch, seq_len, seq_len]
    # v:                 [batch, seq_len, d_model]
    # 输出仍是每个位置一个 d_model 向量：[batch, seq_len, d_model]。
    # 直观理解：每个 token 不再只使用自己的 embedding，而是混合了它关注到的其他 token 信息。
    attention_output = torch.matmul(attention_weights, v)
    assert_shape("attention_output", attention_output, (BATCH_SIZE, SEQ_LEN, D_MODEL))
    show_tensor("attention_output", attention_output)

    # 第一次 residual add：把 attention 分支的输出加回原始 hidden。
    # 这样 attention 学到的是对原表示的“增量修改”，而不是完全替换原表示。
    # 两个张量形状都是 [BATCH_SIZE, SEQ_LEN, D_MODEL]，所以可以逐元素相加。
    hidden_after_attention = hidden + attention_output
    assert_shape("hidden_after_attention", hidden_after_attention, (BATCH_SIZE, SEQ_LEN, D_MODEL))
    show_tensor("hidden_after_attention", hidden_after_attention)

    # FFN 的完整结构也可以写成 nn.Sequential。
    # 这里先保留 Sequential 写法作为对照，但实际展开成三步，
    # 方便分别观察升维、ReLU 激活、降维后的中间张量。
    # feed_forward = nn.Sequential(
    #     nn.Linear(D_MODEL, D_FF),
    #     nn.ReLU(),
    #     nn.Linear(D_FF, D_MODEL),
    # )

    # 第一层把每个位置的向量从 D_MODEL 升维到 D_FF。
    # 第二层再从 D_FF 降回 D_MODEL，保证后续 output_projection 仍能接收 D_MODEL 维输入。
    ffn_up_projection = nn.Linear(D_MODEL, D_FF)
    ffn_down_projection = nn.Linear(D_FF, D_MODEL)

    # FFN 分支也采用 pre-norm：先对第一次 residual 后的结果做 LayerNorm，
    # 再送进 FFN。residual add 时会把 ffn_output 加回 hidden_after_attention。
    ffn_input = ffn_layer_norm(hidden_after_attention)
    assert_shape("ffn_input", ffn_input, (BATCH_SIZE, SEQ_LEN, D_MODEL))
    show_tensor("ffn_input", ffn_input)

    # 对 ffn_input 的最后一维做线性升维。
    # 输入形状是 [BATCH_SIZE, SEQ_LEN, D_MODEL]，
    # 输出形状变成 [BATCH_SIZE, SEQ_LEN, D_FF]。
    # Step 5 的旧版本曾经直接对 attention_output 做 FFN：
    # ffn_hidden = ffn_up_projection(attention_output)
    # Step 6 的 pre-norm 版本改成对 ffn_input 做 FFN。
    ffn_hidden = ffn_up_projection(ffn_input)
    assert_shape("ffn_hidden", ffn_hidden, (BATCH_SIZE, SEQ_LEN, D_FF))
    show_tensor("ffn_hidden", ffn_hidden)

    # ReLU 提供非线性能力：负数会被截断成 0，正数保持不变。
    # 形状不会变化，仍然是 [BATCH_SIZE, SEQ_LEN, D_FF]。
    ffn_activated = F.relu(ffn_hidden)
    assert_shape("ffn_activated", ffn_activated, (BATCH_SIZE, SEQ_LEN, D_FF))
    show_tensor("ffn_activated", ffn_activated)

    # 把 FFN 中间表示降回 D_MODEL。
    # 这样 FFN 可以作为 attention 后面的一个模块，而不会改变主干 hidden size。
    ffn_output = ffn_down_projection(ffn_activated)
    # 这里的 ffn_output 只是 FFN 分支输出，还没有做第二次 residual add。

    # 如果不用展开写法，上面三步可以由这个 Sequential 一次完成：
    # ffn_output = feed_forward(ffn_input)

    assert_shape("ffn_output", ffn_output, (BATCH_SIZE, SEQ_LEN, D_MODEL))
    show_tensor("ffn_output", ffn_output)

    # 第二次 residual add：把 FFN 分支输出加回 hidden_after_attention。
    # 这一步之后才得到一个完整 Transformer block 的输出。
    hidden_after_ffn = hidden_after_attention + ffn_output
    assert_shape("hidden_after_ffn", hidden_after_ffn, (BATCH_SIZE, SEQ_LEN, D_MODEL))
    show_tensor("hidden_after_ffn", hidden_after_ffn)

    # assert_shape("hidden", hidden, (BATCH_SIZE, SEQ_LEN, D_MODEL))
    # show_tensor("hidden", hidden)

    # TODO 4:
    # Create an output projection.
    #
    # Meaning:
    # - Every hidden vector has D_MODEL numbers.
    # - We want one score for each possible token in the vocabulary.
    # - So this layer maps D_MODEL -> VOCAB_SIZE.
    output_projection = nn.Linear(D_MODEL, VOCAB_SIZE)

    # TODO 5:
    # Use output_projection to convert hidden states into logits.
    # 原来的最小语言模型直接把 hidden 投影到词表大小：
    # logits = output_projection(hidden)
    # 后来改成先经过 causal self-attention，再直接把 attention_output 投影到词表：
    # logits = output_projection(attention_output)

    # 再后来加入 FFN，输出层读取的是 ffn_output：
    # logits = output_projection(ffn_output)
    # 现在加入 Add & Norm 后，完整 block 的输出是 hidden_after_ffn。
    # 所以词表投影应该读取 hidden_after_ffn。
    logits = output_projection(hidden_after_ffn)

    assert_shape("logits", logits, (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
    show_tensor("logits", logits)

    # TODO 6:
    # Create targets by hand.
    #
    # Meaning:
    # - targets are the correct token ids that logits should predict.
    # - For now, they only need to be legal token ids.
    #
    # Expected shape:
    #   [BATCH_SIZE, SEQ_LEN] == [2, 4]
    targets = torch.tensor(
        [
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ],
        dtype=torch.long,
    )

    assert_shape("targets", targets, (BATCH_SIZE, SEQ_LEN))
    show_tensor("targets", targets)

    # TODO 7:
    # Compute cross-entropy loss.
    #
    # F.cross_entropy expects:
    #   predictions: [N, VOCAB_SIZE]
    #   answers:     [N]
    #
    # But we have:
    #   logits:  [BATCH_SIZE, SEQ_LEN, VOCAB_SIZE]
    #   targets: [BATCH_SIZE, SEQ_LEN]
    #
    # So flatten the first two dimensions:
    #   [BATCH_SIZE * SEQ_LEN, VOCAB_SIZE]
    #   [BATCH_SIZE * SEQ_LEN]
    loss = F.cross_entropy(
        logits.reshape(BATCH_SIZE * SEQ_LEN, VOCAB_SIZE),
        targets.reshape(BATCH_SIZE * SEQ_LEN),
    )

    print(f"loss: {loss.item():.4f}")
    print("ok: Step 1 pipeline is runnable")
    print()

    # Step 6B:
    # 上面保留的是手写展开版：每个中间张量都能看见，适合学习。
    # 这里把同一条 decoder block 路径封装进 SingleHeadDecoderBlock，再跑一遍。
    # 注意：decoder_block 有自己随机初始化的一套参数，所以 block_loss 不需要等于上面的 loss。
    print("Step 6B: hidden -> SingleHeadDecoderBlock -> block_logits -> block_loss")
    print()

    decoder_block = SingleHeadDecoderBlock()
    block_output = decoder_block(hidden)

    assert_shape("block_output", block_output, (BATCH_SIZE, SEQ_LEN, D_MODEL))
    show_tensor("block_output", block_output)

    # 这里复用已经创建好的 output_projection，把 block_output 投影到词表 logits。
    # 如果以后要做正式模型，可以把这个 projection 也封装进 MiniTransformerLM。
    block_logits = output_projection(block_output)

    assert_shape("block_logits", block_logits, (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
    show_tensor("block_logits", block_logits)

    block_loss = F.cross_entropy(
        block_logits.reshape(BATCH_SIZE * SEQ_LEN, VOCAB_SIZE),
        targets.reshape(BATCH_SIZE * SEQ_LEN),
    )

    print(f"block_loss: {block_loss.item():.4f}")
    print("ok: SingleHeadDecoderBlock is runnable")
    print()

    # Step 8B:
    # 保留上面的 SingleHeadDecoderBlock 作为“单头封装版”学习痕迹。
    # 这里额外跑 MultiHeadDecoderBlock，验证多头拆分、每个 head 独立 attention、
    # concat 回 D_MODEL、再接 FFN 的完整路径。
    print("Step 8B: hidden -> MultiHeadDecoderBlock -> multihead_logits -> multihead_loss")
    print()

    multihead_decoder_block = MultiHeadDecoderBlock()
    multihead_output = multihead_decoder_block(hidden)

    assert_shape("multihead_output", multihead_output, (BATCH_SIZE, SEQ_LEN, D_MODEL))
    show_tensor("multihead_output", multihead_output)

    # 继续复用 output_projection，把多头 block 的输出投影成词表 logits。
    # 多头 block 有自己随机初始化的参数，所以 multihead_loss 也不需要等于前面的 loss。
    multihead_logits = output_projection(multihead_output)

    assert_shape("multihead_logits", multihead_logits, (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
    show_tensor("multihead_logits", multihead_logits)

    multihead_loss = F.cross_entropy(
        multihead_logits.reshape(BATCH_SIZE * SEQ_LEN, VOCAB_SIZE),
        targets.reshape(BATCH_SIZE * SEQ_LEN),
    )

    print(f"multihead_loss: {multihead_loss.item():.4f}")
    print("ok: MultiHeadDecoderBlock is runnable")
    print()

    # TODO 8-9:
    # Now package the same idea into TinyTokenModel.
    #
    # This is not new math. It is the same path:
    #   token_ids -> token_embedding -> hidden -> output_projection -> logits
    #
    # The model has its own randomly initialized parameters, so model_loss does
    # not need to equal the manual loss above. Only the shape and runnable path
    # matter at this step.
    print("Step 1B: token_ids -> TinyTokenModel -> model_logits -> model_loss")
    print()

    model = TinyTokenModel()
    model_logits = model(token_ids)

    assert_shape("model_logits", model_logits, (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
    show_tensor("model_logits", model_logits)

    model_loss = F.cross_entropy(
        model_logits.reshape(BATCH_SIZE * SEQ_LEN, VOCAB_SIZE),
        targets.reshape(BATCH_SIZE * SEQ_LEN),
    )

    print(f"model_loss: {model_loss.item():.4f}")
    print("ok: TinyTokenModel is runnable")


class TinyTokenModel(nn.Module):
    """
    Use this only after TODO 1-7 work in main().

    The purpose of this class is not new math. It just packages the two layers
    you already tested by hand:

        token_ids -> token_embedding -> hidden -> output_projection -> logits
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO 8:
        # Move your working token_embedding and output_projection layers here.
        self.token_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.output_projection = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, token_ids: TokenIds) -> Logits:
        # TODO 9:
        # Move your working forward flow here:
        # token_ids -> hidden -> logits
        hidden: HiddenStates = self.token_embedding(token_ids)
        assert_shape("hidden", hidden, (BATCH_SIZE, SEQ_LEN, D_MODEL))

        logits: Logits = self.output_projection(hidden)
        assert_shape("logits", logits, (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
        return logits


if __name__ == "__main__":
    main()
