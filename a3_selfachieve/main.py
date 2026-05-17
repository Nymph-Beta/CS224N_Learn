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
ValueStates = Tensor  # shape: [BATCH_SIZE, SEQ_LEN, D_MODEL
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
    logits = output_projection(hidden)

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
