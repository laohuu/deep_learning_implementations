import torch
from torch import nn


class GroupNorm(nn.Module):
    """
    ## Group Normalization Layer
    """

    def __init__(self, groups: int, channels: int, *,
                 eps: float = 1e-5, affine: bool = True):
        """
        * `groups` is the number of groups the features are divided into
        * `channels` is the number of features in the input
        * `affine` is whether to scale and shift the normalized value
        """
        super().__init__()

        assert channels % groups == 0, "Number of channels should be evenly divisible by the number of groups"
        self.groups = groups
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

        x = x.view(batch_size, self.groups, -1)

        mean = x.mean(dim=[-1], keepdim=True)
        mean_x2 = (x ** 2).mean(dim=[-1], keepdim=True)

        var = mean_x2 - mean ** 2

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x_norm = x_norm.view(batch_size, self.channels, -1)
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)

        return x_norm.view(x_shape)


def _test():
    x = torch.zeros([2, 6, 2, 4])
    print(x.shape)
    bn = GroupNorm(2, 6)

    x = bn(x)
    print(x.shape)


if __name__ == '__main__':
    _test()
