import torch
from torch import nn


class BatchNorm(nn.Module):
    """
    ## Batch Normalization Layer
    """

    def __init__(self, channels: int, *,
                 eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True):
        """
        * `channels` is the number of features in the input
        * `momentum` is the momentum in taking the exponential moving average
        * `affine` is whether to scale and shift the normalized value
        * `track_running_stats` is whether to calculate the moving averages or mean and variance
        """
        super().__init__()

        self.channels = channels

        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.scale = nn.Parameter(torch.ones(channels))
            self.shift = nn.Parameter(torch.zeros(channels))

        if self.track_running_stats:
            self.register_buffer('exp_mean', torch.zeros(channels))
            self.register_buffer('exp_var', torch.ones(channels))

    def forward(self, x: torch.Tensor):
        """
        `x` is a tensor of shape `[batch_size, channels, *]`.
        """

        x_shape = x.shape
        batch_size = x_shape[0]
        assert self.channels == x.shape[1]

        # Reshape into `[batch_size, channels, n]`
        x = x.view(batch_size, self.channels, -1)

        # We will calculate the mini-batch mean and variance
        # if we are in training mode or if we have not tracked exponential moving averages
        if self.training or not self.track_running_stats:
            mean = x.mean(dim=[0, 2])
            mean_x2 = (x ** 2).mean(dim=[0, 2])

            var = mean_x2 - mean ** 2

            # Update exponential moving averages
            if self.training and self.track_running_stats:
                self.exp_mean = (1 - self.momentum) * self.exp_mean + self.momentum * mean
                self.exp_var = (1 - self.momentum) * self.exp_var + self.momentum * var
        # Use exponential moving averages as estimates
        else:
            mean = self.exp_mean
            var = self.exp_var

        # Normalize
        x_norm = (x - mean.view(1, -1, 1)) / torch.sqrt(var + self.eps).view(1, -1, 1)
        # Scale and shift
        if self.affine:
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)

        return x_norm.view(x_shape)


def _test():
    x = torch.zeros([2, 3, 2, 4])
    print(x.shape)
    bn = BatchNorm(3)

    x = bn(x)
    print(x.shape)
    print(bn.exp_var.shape)


if __name__ == '__main__':
    _test()
