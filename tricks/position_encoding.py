import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def create_1d_absolute_sin_cos_embedding(n_pos_vector, d_model):
    # n_pos_vector = torch.arange(n)
    assert d_model % 2 == 0, "wrong"
    position_embedding = torch.zeros(n_pos_vector.numel(), d_model)

    position = torch.arange(0, n_pos_vector.numel()).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))

    position_embedding[:, 0::2] = torch.sin(position * div_term)
    position_embedding[:, 1::2] = torch.cos(position * div_term)

    return position_embedding


def create_1d_absolute_trainable_embedding(n_pos_vector, d_model):
    position_embedding = nn.Embedding(n_pos_vector.numel(), dim)
    nn.init.constant_(position_embedding.weight, 0.)

    return position_embedding


# swin transformer
# relative position bias
def create_2d_relative_bias_trainable_embeddings(n_head, height, width, dim):
    # width: 5, [0, 1, 2, 3, 4] bias=[-4,4] , 2*width-1
    # height: 5, [0, 1, 2, 3, 4] bias=[-4,4] , 2*height-1
    position_embedding = nn.Embedding((2 * width - 1) * (2 * height - 1), n_head)
    nn.init.constant_(position_embedding.weight, 0.)

    def get_relative_position_index(height, width):
        coords = torch.stack(torch.meshgrid(torch.arange(height), torch.arange(width)))
        coords_flatten = torch.flatten(coords, 1)  # 【2，height*width】

        relative_coords_bias = coords_flatten[:, None, :] - coords_flatten[:, :, None]
        relative_coords_bias[0, :, :] += height - 1
        relative_coords_bias[1, :, :] += width - 1

        # A:2d , B:1d, B[i*cols+j] = A[i,j]
        relative_coords_bias[0, :, :] *= relative_coords_bias[1, :, :].max() + 1

        return relative_coords_bias.sum(0)  # [height*width,height*width]

    relative_position_bias = get_relative_position_index(height, width)
    bias_embedding = position_embedding(torch.flatten(relative_position_bias)).reshape(height * width, height * width,
                                                                                       n_head)
    bias_embedding = torch.unsqueeze(bias_embedding.permute(2, 0, 1), dim=0)  # [1,n_heads,height*width, height * width]
    return bias_embedding


# mae
# 2d absolute constant sincos embedding

def create_2d_absolute_sincos_embeddings(height, width, dim):
    assert dim % 4 == 0, "wrong"
    position_embedding = torch.zeros(height * width, dim)
    coords = torch.stack(torch.meshgrid(torch.arange(height), torch.arange(width)))

    height_embedding = create_1d_absolute_sin_cos_embedding(torch.flatten(coords[0]), dim // 2)
    width_embedding = create_1d_absolute_sin_cos_embedding(torch.flatten(coords[1]), dim // 2)

    position_embedding[:, :dim // 2] = height_embedding
    position_embedding[:, dim // 2:] = width_embedding

    return position_embedding


if __name__ == "__main__":
    n_pos = 4
    dim = 4
    n_pos_vector = torch.arange(n_pos)
    print(create_1d_absolute_sin_cos_embedding(n_pos_vector, dim))
