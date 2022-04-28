# BatchNorm/LayerNorm/InsNorm/GroupNorm/WeightNorm
import torch
import torch.nn as nn
import torch.nn.functional as F

# Batch Norm
# NLP: N*L*C -> C
# CV: N*C*H*W - > C

# Layer Norm
# NLP: N,L,C -> N,L
# CV: N,C,H,W -> N

# INS NORM
# NLP: N,L,C -> N,C
# CV: N,C,H,W -> N,C

# GROUP NORM
# NLP: N,G,L,C//G -> N,G
# CV: N,G,C//G,H,W -> N,G

batch_size = 2
time_steps = 3
embedding_dim = 4
eps = 1e-6

# 检测一维实现
input = torch.randn(size=(batch_size, time_steps, embedding_dim))  # N*L*C

# BatchNorm
pytorch_batch_norm_op = nn.BatchNorm1d(embedding_dim, affine=False)
pytorch_batch_norm_output = pytorch_batch_norm_op(input.transpose(-1, -2)).transpose(-1, -2)

# 手写BatchNorm
bn_mean = torch.mean(input, dim=(0, 1), keepdim=True)
bn_std = torch.std(input, dim=(0, 1), unbiased=False, keepdim=True)
verify_bn_y = (input - bn_mean) / (bn_std + eps)

print(torch.allclose(verify_bn_y, pytorch_batch_norm_output, rtol=1e-4))

# LayerNorm
pytorch_layer_norm_op = nn.LayerNorm(embedding_dim, elementwise_affine=False)
pytorch_layer_norm_output = pytorch_layer_norm_op(input)

# 手写LayerNorm
layer_mean = torch.mean(input, dim=-1, keepdim=True)
layer_std = torch.std(input, dim=-1, unbiased=False, keepdim=True)
verify_ln_y = (input - layer_mean) / (layer_std + eps)

print(torch.allclose(verify_ln_y, pytorch_layer_norm_output, rtol=1e-4))

# InsNorm :所有时刻做均值，一般用于风格迁移
pytorch_ins_norm_op = nn.InstanceNorm1d(embedding_dim)
pytorch_ins_norm_output = pytorch_ins_norm_op(input.transpose(-1, -2)).transpose(-1, -2)

# 手写InsNorm
ins_mean = torch.mean(input, dim=1, keepdim=True)
ins_std = torch.std(input, dim=1, unbiased=False, keepdim=True)
verify_ins_norm_y = (input - ins_mean) / (ins_std + eps)

print(torch.allclose(verify_ins_norm_y, pytorch_ins_norm_output, rtol=1e-4))

# GroupNorm：per sample per group
num_groups = 2
pytorch_group_norm_op = nn.GroupNorm(num_groups, embedding_dim, affine=False)
pytorch_group_norm_output = pytorch_group_norm_op(input.transpose(-1, -2)).transpose(-1, -2)

# 手写GroupNorm
group_input = torch.split(input, split_size_or_sections=embedding_dim // num_groups, dim=-1)
result = []
for g_inout in group_input:
    group_mean = torch.mean(g_inout, dim=(1, 2), keepdim=True)
    group_std = torch.std(g_inout, dim=(1, 2), unbiased=False, keepdim=True)
    verify_group_norm_y = (g_inout - group_mean) / (group_std + eps)
    result.append(verify_group_norm_y)

verify_group_norm_y = torch.cat(result, dim=-1)

print(torch.allclose(verify_group_norm_y, pytorch_group_norm_output, rtol=1e-4))

# WeightNorm：
linear_module = nn.Linear(embedding_dim, 3, bias=False)
wn_module = nn.utils.weight_norm(linear_module)
pytorch_weight_norm_output = wn_module(input)

# 手写WeightNorm
linear_weight_v = linear_module.weight / linear_module.weight.norm(dim=1, keepdim=True)
linear_weight_g = wn_module.weight_g
verify_weight_norm_y = input @ linear_weight_v.T * linear_weight_g.T

print(torch.allclose(verify_weight_norm_y, pytorch_weight_norm_output, rtol=1e-4))

batch_size = 10
channels = 3
H = 24
W = 24

print("# 二维实现")
# 二维实现
input_image = torch.randn(size=(batch_size, channels, H, W))
pytorch_2d_bn = nn.BatchNorm2d(channels, affine=False)
pytorch_2d_bn_output = pytorch_2d_bn(input_image)

# 手写BatchNorm
bn_mean = torch.mean(input_image, dim=(0, 2, 3), keepdim=True)
bn_std = torch.std(input_image, dim=(0, 2, 3), unbiased=False, keepdim=True)
verify_bn_y = (input_image - bn_mean) / (bn_std + eps)

print(torch.allclose(verify_bn_y, pytorch_2d_bn_output))

# layer norm
pytorch_2d_ln = nn.LayerNorm([channels, H, W], elementwise_affine=False)
pytorch_2d_ln_output = pytorch_2d_ln(input_image)

# 手写layer norm
ln_mean = torch.mean(input_image, dim=(1, 2, 3), keepdim=True)
ln_std = torch.std(input_image, dim=(1, 2, 3), unbiased=False, keepdim=True)
verify_ln_y = (input_image - ln_mean) / (ln_std + eps)

print(torch.allclose(verify_ln_y, pytorch_2d_ln_output))

# ins norm
pytorch_2d_in = nn.InstanceNorm2d(channels, affine=False)
pytorch_2d_in_output = pytorch_2d_in(input_image)

# 手写ins norm
in_mean = torch.mean(input_image, dim=(2, 3), keepdim=True)
in_std = torch.std(input_image, dim=(2, 3), unbiased=False, keepdim=True)
verify_in_y = (input_image - in_mean) / (in_std + eps)

print(torch.allclose(verify_in_y, pytorch_2d_in_output))

# group norm
num_groups = 3
pytorch_2d_gn = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
pytorch_2d_gn_output = pytorch_2d_gn(input_image)

# 手写group norm
input_image_gn = torch.reshape(input_image, shape=(batch_size, num_groups, -1, H, W))
gn_mean = torch.mean(input_image_gn, dim=(2, 3, 4), keepdim=True)
gn_std = torch.std(input_image_gn, dim=(2, 3, 4), unbiased=False, keepdim=True)
verify_gn_y = (input_image_gn - gn_mean) / (gn_std + eps)

verify_gn_y = torch.reshape(verify_gn_y, shape=(batch_size, channels, H, W))
print(torch.allclose(verify_gn_y, pytorch_2d_gn_output, rtol=1e-4))
