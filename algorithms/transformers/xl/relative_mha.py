"""
# Relative Multi-Headed Attention

This is an implementation of relative multi-headed attention from paper
[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://papers.labml.ai/paper/1901.02860)
in [PyTorch](https://pytorch.org).
"""

import torch
from torch import nn

from algorithms.transformers.mha import MultiHeadAttention


def shift_right(x: torch.Tensor):
    # Concatenate a column of zeros
    zero_pad = x.new_zeros(x.shape[0], 1, *x.shape[2:])
    x_padded = torch.cat([x, zero_pad], dim=1)

    # Reshape and remove excess elements from the end
    x_padded = x_padded.view(x.shape[1] + 1, x.shape[0], *x.shape[2:])
    x = x_padded[:-1].view_as(x)

    return x


class RelativeMultiHeadAttention(MultiHeadAttention):
    """
    We override [Multi-Head Attention] module so we only need to
    write the `get_scores` method.
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        # The linear transformations do not need a bias since we
        # explicitly include it when calculating scores.
        # However having a bias for `value` might make sense.
        super().__init__(heads, d_model, dropout_prob, bias=False)

        # Number of relative positions
        self.P = 2 ** 12

        self.key_pos_embeddings = nn.Parameter(torch.zeros((self.P * 2, heads, self.d_k)), requires_grad=True)
        self.key_pos_bias = nn.Parameter(torch.zeros((self.P * 2, heads)), requires_grad=True)
        self.query_pos_bias = nn.Parameter(torch.zeros((heads, self.d_k)), requires_grad=True)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        r"""
        ### Get relative attention scores
        """
        key_pos_emb = self.key_pos_embeddings[self.P - key.shape[0]:self.P + query.shape[0]]
        key_pos_bias = self.key_pos_bias[self.P - key.shape[0]:self.P + query.shape[0]]
        query_pos_bias = self.query_pos_bias[None, None, :, :]

        ac = torch.einsum('ibhd,jbhd->ijbh', query + query_pos_bias, key)
        b = torch.einsum('ibhd,jhd->ijbh', query, key_pos_emb)

        d = key_pos_bias[None, :, None, :]

        bd = shift_right(b + d)
        bd = bd[:, -key.shape[0]:]
        return ac + bd


def _test_shift_right():
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(x)
    print(shift_right(x))

    x = torch.arange(1, 6)[None, :, None, None].repeat(5, 1, 1, 1)
    print(x[:, :, 0, 0])
    print(shift_right(x)[:, :, 0, 0])

    x = torch.arange(1, 6)[None, :, None, None].repeat(3, 1, 1, 1)
    print(x[:, :, 0, 0])
    print(shift_right(x)[:, :, 0, 0])


if __name__ == '__main__':
    _test_shift_right()
