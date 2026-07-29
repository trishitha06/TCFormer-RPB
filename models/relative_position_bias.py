import torch
import torch.nn as nn


class RelativePositionBias(nn.Module):
    """
    Learnable Relative Position Bias (RPB)

    Generates a head-specific relative position bias matrix
    that is added to the attention logits BEFORE softmax.

    Output:
        (num_heads, seq_len, seq_len)

    Compatible with:
        - Multi-Head Attention
        - Grouped Query Attention
        - Rotary Position Embedding
    """

    def __init__(
        self,
        num_heads: int,
        max_position: int = 512,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.max_position = max_position

        # Number of possible relative positions
        #
        # [-max_position+1, ..., 0, ..., max_position-1]
        #
        self.relative_bias_table = nn.Parameter(
            torch.zeros(
                2 * max_position - 1,
                num_heads,
            )
        )

        nn.init.trunc_normal_(
            self.relative_bias_table,
            std=0.02,
        )

        self.register_buffer(
            "relative_position_index",
            self._build_index(max_position),
            persistent=False,
        )

    @staticmethod
    def _build_index(max_position):

        coords = torch.arange(max_position)

        relative_coords = (
            coords[:, None] - coords[None, :]
        )

        relative_coords += max_position - 1

        return relative_coords.long()

    def forward(self, seq_len):

        if seq_len > self.max_position:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds "
                f"maximum position ({self.max_position})."
            )

        index = self.relative_position_index[
            :seq_len,
            :seq_len,
        ]

        bias = self.relative_bias_table[index]

        # (T,T,H)
        bias = bias.permute(2, 0, 1)

        # (H,T,T)
        return bias