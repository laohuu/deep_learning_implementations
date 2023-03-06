"""
# Position-wise Feed-Forward Network (FFN)

This is a [PyTorch](https://pytorch.org)  implementation
of position-wise feedforward network used in transformer.

Sometimes the
GELU (Gaussian Error Linear Unit) activation is also used instead of ReLU.
### Gated Linear Units
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    ## FFN module
    """

    def __init__(self, d_model: int, d_ff: int,
                 dropout: float = 0.1,
                 activation=nn.ReLU(),
                 is_gated: bool = False,
                 bias1: bool = True,
                 bias2: bool = True,
                 bias_gate: bool = True):
        """
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability for the hidden layer
        * `is_gated` specifies whether the hidden layer is gated
        * `bias1` specified whether the first fully connected layer should have a learnable bias
        * `bias2` specified whether the second fully connected layer should have a learnable bias
        * `bias_gate` specified whether the fully connected layer for the gate should have a learnable bias
        """
        super().__init__()

        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        # Hidden layer dropout
        self.dropout = nn.Dropout(dropout)

        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            # If there is a gate the linear layer to transform inputs to
            # be multiplied by the gate
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: torch.Tensor):

        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_v(x)
        else:
            x = g
        # Apply dropout
        x = self.dropout(x)

        return self.layer2(x)
