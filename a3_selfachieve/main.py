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

    # TODO 3:
    # Use token_embedding to convert token_ids into hidden states.
    hidden = token_embedding(token_ids)

    # 自注意力的第一步是把同一份 hidden states 映射成三种不同角色：
    # - q/query：当前位置主动发出的“查询”，用来问自己该关注哪些位置。
    # - k/key：每个位置提供的“索引标签”，用来和 query 计算匹配程度。
    # - v/value：每个位置真正携带的信息内容，最后会被加权汇总。
    # 这里先用单头注意力，所以 q/k/v 的最后一维仍保持 D_MODEL 不变。
    q_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)
    k_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)
    v_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)

    # 对每个 token 的 hidden vector 分别做线性变换。
    # 输入 hidden 的形状是 [batch, seq_len, d_model]；
    # Linear 会作用在最后一维，因此 q/k/v 的形状仍是 [batch, seq_len, d_model]。
    q = q_projection(hidden)
    k = k_projection(hidden)
    v = v_projection(hidden)

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

    # softmax 在最后一维上做归一化，也就是让每个 query 对所有 key 的分数
    # 变成一行概率分布。归一化后，每一行的和应该接近 1。
    attention_weights = F.softmax(scores, dim=-1)
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

    # 用注意力权重对 value 做加权求和。
    # attention_weights: [batch, seq_len, seq_len]
    # v:                 [batch, seq_len, d_model]
    # 输出仍是每个位置一个 d_model 向量：[batch, seq_len, d_model]。
    # 直观理解：每个 token 不再只使用自己的 embedding，而是混合了它关注到的其他 token 信息。
    attention_output = torch.matmul(attention_weights, v)
    assert_shape("attention_output", attention_output, (BATCH_SIZE, SEQ_LEN, D_MODEL))
    show_tensor("attention_output", attention_output)

    assert_shape("hidden", hidden, (BATCH_SIZE, SEQ_LEN, D_MODEL))
    show_tensor("hidden", hidden)

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
    # 现在中间多了一层自注意力，所以输出层读取的是 attention_output。
    # 这样每个位置的预测会基于注意力混合后的上下文表示，而不是只基于原始 embedding。
    logits = output_projection(attention_output)

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
