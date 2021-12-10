# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        # 用于cuda实现
        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # 采样点的坐标偏移，每个query在每个注意力头和每个特征层都需要采样n_points个。
        # 由于x,y坐标都有对应的偏移量，所以还要*2
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        # 每个query对应的所有采样点的注意力权重
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        # 线性变换得到value
        self.value_proj = nn.Linear(d_model, d_model)
        # 最后经过这个线性变换得到输出结果
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        """初始化偏移量预测的偏置（bias），使得初始偏移位置犹如不同大小的方形卷积核组合。"""
        # (8,) [0,pi/4,pi/2,3*pi/2, ... ,7*pi/4]
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        # (8,2)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # grid_init / grid_init.abs().max(-1, keepdim=True)[0] 这一步计算得到8个头对应的坐标偏移。
        # (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)
        # 然后repeat给所有特征层和所有采样点
        # (8,4,4,2)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2). \
            repeat(1, self.n_levels, self.n_points, 1)
        # 同一个也整层中不同采样点的坐标偏移肯定不能一样，所以这里做了处理。对于第i个采样点，在8个头部和所有特征层中，其坐标偏移是：
        # (i,0), (i,i), (i,0), (-i,i), (-i,0), (-i,-i), (0,-i) (i,-i)
        # 从图形视觉上来看，形成的偏移位置相当于是3*3, 5*5, 7*7, 9*9 正方形卷积核（除去中心，中心是参考点本身）
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        # 这里取消了梯度，只是借助nn.Parameter把数值设置进去。
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)

        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)

        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index,
                input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)

        Multi-Scale Deformable Attention主要做以下事情：
        1. 将输入input_flatten通过变换矩阵得到value，同时将padding的部分用0填充；
            （对于Encoder来说就是由backbone输出的特征图变换而来，对于decoder就是encoder输出的memory）
        2. 将query分别通过两个全连接层得到采样点对应的坐标便宜和注意力权重（注意力权重会进行归一化）
            （对于encoder来说就是特征图本身加上position embedding 和 scale-level embedding的结果，
                对于decoder来说就是self-attention的输出加上position embedding的结果，
                2-stage时这个position embedding是由encoder预测的top-k proposal boxes进行position embedding得来；
                而1-stage时是预设的query embedding分别通过两个全连接层得到采样点对应的坐标偏移和注意力权重（注意力权重会进行归一化）
        3. 根据参考点坐标和预测的坐标偏移得到采样点的坐标。
        4. 由采样点坐标在value中插值采样出对应特征的向量，然后施加注意力权重，最后将这个结果经过全连接层得到输出结果。
        """

        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        # 这个值需要是所有特征层特征点的数量）
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        # (N,Len_in,d_model=256)
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            # 将原图padding的部分用0填充
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        # (N, Len_in, 8, 64)拆分成8个注意力头部对应的维度。
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        # (N, Len_q, 8, 4, 4, 2)预测采样点的坐标偏移
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        # (N, Len_q, 8, 4*4) 预测采样点对应的注意力权重
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        # (N, Len_q, 8, 4*4) 每个query在每个注意力头部内，每个特征层都采样4个特征点，即16个采样点，这16个对应的权重进行归一化
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        '''下面是计算采样点的位置。(以下参考点的坐标已归一化）'''
        # sampling_locations: N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            # (4,2)其中每个是(w,h)
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # 对坐标偏移量使用对应层特征图的宽高进行归一化然后加在参考点坐标上得到采样点坐标
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            # 最后一维中的前两个是中心坐标xy，后两个坐标是宽高wh。
            # 由于初始化时，offset的在-K~K(K=n_points)范围，所以这里除以n_points相当于归一化到0-1.
            # 然后乘以宽和高的一半，加上参考点的中心坐标，这样就使得偏移后的采样点位于proposal Bbox内，
            # 相当于对采样范围进行了约束，减少了搜索空间
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        # 注意力权重有了，采样点位置有了，最后就是计算结果了。
        # 根据采样点位置才出对应的value，并且施加预测出来的注意力权重（和value进行weighted sum）
        # (N, Len_in, 256)
        # 注： 实质调用的是基于CUDA实现的版本，需要编译(pytorch实现的性能较差)。
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights,
            self.im2col_step)
        # (N, Len_in, 256)
        output = self.output_proj(output)

        return output
