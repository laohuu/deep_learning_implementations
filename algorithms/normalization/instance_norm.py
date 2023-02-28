import torch
from torch import nn


class InstanceNorm(nn.Module):
    """
    ## Instance Normalization Layer
    """

    def __init__(self, channels: int, *,
                 eps: float = 1e-5, affine: bool = True):
        """
        * `channels` is the number of features in the input
        * `affine` is whether to scale and shift the normalized value
        """
        super().__init__()

        self.channels = channels

        self.eps = eps
        self.affine = affine

        if self.affine:
            self.scale = nn.Parameter(torch.ones(channels))
            self.shift = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor):
        """
        `x` is a tensor of shape `[batch_size, channels, *]`.
        `*` denotes any number of (possibly 0) dimensions.
         For example, in an image (2D) convolution this will be
        `[batch_size, channels, height, width]`
        """
        x_shape = x.shape
        batch_size = x_shape[0]
        assert self.channels == x.shape[1]

        # Reshape into [batch_size, channels, n]
        x = x.view(batch_size, self.channels, -1)

        mean = x.mean(dim=[-1], keepdim=True)
        mean_x2 = (x ** 2).mean(dim=[-1], keepdim=True)

        var = mean_x2 - mean ** 2

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.view(batch_size, self.channels, -1)

        if self.affine:
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)

        return x_norm.view(x_shape)


def _test():
    x = torch.zeros([2, 6, 2, 4])
    print(x.shape)
    bn = InstanceNorm(6)

    x = bn(x)
    print(x.shape)


if __name__ == '__main__':
    _test()
