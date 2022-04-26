import math

import torch
import torch.nn as nn
import torch.nn.functional as F

input = torch.randn(size=(5, 5))
kernel = torch.randn(size=(3, 3))
bias = torch.randn(1)


# 原始的遍历运算来实现二维卷积
def matrix_sum_for_conv2d(input, kernel, bias=0, stride=1, padding=0):
    if padding > 0:
        input = F.pad(input, (padding, padding, padding, padding))

    input_h, input_w = input.size()
    kernel_h, kernel_w = kernel.size()

    output_h = math.floor((input_h - kernel_h) / stride) + 1
    output_w = math.floor((input_w - kernel_w) / stride) + 1

    output = torch.zeros(size=(output_h, output_w))
    for i in range(0, input_h - kernel_h + 1, stride):  # 对高度遍历
        for j in range(0, input_w - kernel_w + 1, stride):  # 对宽度遍历
            region = input[i:i + kernel_h, j:j + kernel_w]
            output[int(i / stride), int(j / stride)] = torch.sum(region * kernel) + bias

    return output


# 矩阵运算计算卷积
def matrix_multiplication_for_conv2d(input, kernel, bias=0, stride=1, padding=0):
    if padding > 0:
        input = F.pad(input, (padding, padding, padding, padding))

    input_h, input_w = input.size()
    kernel_h, kernel_w = kernel.size()

    output_h = (math.floor((input_h - kernel_h) / stride) + 1)
    output_w = (math.floor((input_w - kernel_w) / stride) + 1)

    output = torch.zeros(size=(output_h, output_w))

    region_matrix = torch.zeros(output.numel(), kernel.numel())
    kernel_matrix = kernel.reshape(shape=(kernel.numel(), 1))
    cnt_index = 0
    for i in range(0, input_h - kernel_h + 1, stride):  # 对高度遍历
        for j in range(0, input_w - kernel_w + 1, stride):  # 对宽度遍历
            region = input[i:i + kernel_h, j:j + kernel_w]
            region_vector = torch.flatten(region)
            region_matrix[cnt_index] = region_vector
            cnt_index = cnt_index + 1

    output_matrix = torch.matmul(region_matrix, kernel_matrix)
    output = output_matrix.reshape(output_h, output_w) + bias
    return output


pytorch_api_conv_output = F.conv2d(input.reshape(1, 1, input.shape[0], input.shape[1]),
                                   kernel.reshape(1, 1, kernel.shape[0], kernel.shape[1]),
                                   padding=1, bias=bias, stride=1)
matrix_sum_for_conv_output = matrix_sum_for_conv2d(input, kernel, padding=1, bias=bias, stride=1)
mat_mul_conv_output = matrix_multiplication_for_conv2d(input, kernel, padding=1, bias=bias, stride=1)

print(torch.allclose(pytorch_api_conv_output, matrix_sum_for_conv_output))
print(torch.allclose(pytorch_api_conv_output, mat_mul_conv_output))


# 考虑batch size 和 channel维度
def matrix_sum_for_conv2d_full(input, kernel, bias=0, stride=1, padding=0):
    if padding > 0:
        input = F.pad(input, (padding, padding, padding, padding))

    batch_size, input_channel, input_h, input_w = input.size()
    output_channel, input_channel, kernel_h, kernel_w = kernel.size()

    output_h = math.floor((input_h - kernel_h) / stride) + 1
    output_w = math.floor((input_w - kernel_w) / stride) + 1

    output = torch.zeros(size=(batch_size, output_channel, output_h, output_w))
    for ind in range(batch_size):
        for oc in range(output_channel):
            for ic in range(input_channel):
                for i in range(0, input_h - kernel_h + 1, stride):  # 对高度遍历
                    for j in range(0, input_w - kernel_w + 1, stride):  # 对宽度遍历
                        region = input[ind, ic, i:i + kernel_h, j:j + kernel_w]
                        output[ind, oc, int(i / stride), int(j / stride)] += torch.sum(region * kernel[oc, ic])
            output[ind, oc] += bias[oc]

    return output


input = torch.randn(size=(2, 2, 5, 5))
kernel = torch.randn(size=(3, 2, 3, 3))
bias = torch.randn(3)

pytorch_api_full = F.conv2d(input, kernel, bias, padding=1, stride=2)
sum_api_full = matrix_sum_for_conv2d_full(input, kernel, bias=bias, padding=1, stride=2)
print(torch.allclose(pytorch_api_full, sum_api_full))

# 转置卷积
