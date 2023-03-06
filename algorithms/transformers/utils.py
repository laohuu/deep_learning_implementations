"""
# Utilities for Transformer
"""

import torch
from typing import Any, TypeVar, Iterator, Iterable, Generic
import torch.nn as nn
import copy


def clone_module_list(module: nn.Module, n: int) -> nn.ModuleList[nn.Module]:
    """
    ## Clone Module

    Make a `nn.ModuleList` with clones of a given module
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(seq_len):
    """
    ## Subsequent mask to mask out data from future (subsequent) time steps
    """
    mask = torch.tril(torch.ones(seq_len, seq_len)).to(torch.bool).unsqueeze(-1)
    return mask


def _subsequent_mask():
    print(subsequent_mask(10)[:, :, 0])


if __name__ == '__main__':
    _subsequent_mask()
