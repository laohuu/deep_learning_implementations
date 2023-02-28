import torch


def weight_standardization(weight: torch.Tensor, eps: float):
    """
    ## Weight Standardization
    """

    c_out, c_in, *kernel_shape = weight.shape

    weight = weight.view(c_out, -1)

    var, mean = torch.var_mean(weight, dim=1, keepdim=True)
    # Normalize
    weight = (weight - mean) / (torch.sqrt(var + eps))

    return weight.view(c_out, c_in, *kernel_shape)
