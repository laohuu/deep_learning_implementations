from typing import Union, List

import torch
from torch import nn, Size


class LayerNorm(nn.Module):
    """
    ## Layer Normalization
    """

    def __init__(self, normalized_shape: Union[int, List[int], Size], *,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True):
        """
        * `normalized_shape` S is the shape of the elements (except the batch).
        * `elementwise_affine` is whether to scale and shift the normalized value
        """
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        elif isinstance(normalized_shape, list):
            normalized_shape = torch.Size(normalized_shape)
        assert isinstance(normalized_shape, torch.Size)

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.gain = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor):
        """
        `x` is a tensor of shape `[*, S[0], S[1], ..., S[n]]`.
        `*` could be any number of dimensions.
         For example, in an NLP task this will be
        `[seq_len, batch_size, features]`
        """
        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]

        # The dimensions to calculate the mean and variance on
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]

        # Calculate the mean of all elements;
        mean = x.mean(dim=dims, keepdim=True)
        mean_x2 = (x ** 2).mean(dim=dims, keepdim=True)

        var = mean_x2 - mean ** 2

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            x_norm = self.gain * x_norm + self.bias

        return x_norm


def _test():
    x = torch.zeros([2, 3, 2, 4])
    print(x.shape)
    ln = LayerNorm(x.shape[2:])

    x = ln(x)
    print(x.shape)
    print(ln.gain.shape)


if __name__ == '__main__':
    _test()
