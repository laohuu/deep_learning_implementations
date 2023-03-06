"""
# Transformer Auto-Regression Experiment

This trains a simple transformer introduced in [Attention Is All You Need](https://papers.labml.ai/paper/1706.03762)
on an NLP auto-regression task (with Tiny Shakespeare dataset).
"""

import torch
from torch import nn

from algorithms.transformers.models import Encoder
from algorithms.transformers.utils import subsequent_mask
from torchtext import data, datasets


class AutoregressiveTransformer(nn.Module):
    def __init__(self, encoder: Encoder, src_embed: nn.Module, generator: nn.Module):
        """
        * `encoder` is the transformer
        * `src_embed` is the token [embedding module (with positional encodings)]
        * `generator` is the final fully connected layer
        """
        super().__init__()
        self.src_embed = src_embed
        self.encoder = encoder
        self.generator = generator

        # The mask will be initialized on the first call
        self.mask = None

    def forward(self, x: torch.Tensor):
        # Create subsequent mask if mask is not initialized
        # or if the size of the mask is different
        if self.mask is None or self.mask.size(0) != len(x):
            # Subsequent mask, will mask out tokens from seeing future tokens
            self.mask = subsequent_mask(len(x)).to(x.device)
        # Get the token embeddings with positional encodings
        x = self.src_embed(x)
        # Transformer encoder
        x = self.encoder(x, self.mask)
        # Get logits
        x = self.generator(x)

        # Return results
        # (second value is for state, since our trainer is used with RNNs also)
        return x, None


def main():
    pass


if __name__ == '__main__':
    main()
