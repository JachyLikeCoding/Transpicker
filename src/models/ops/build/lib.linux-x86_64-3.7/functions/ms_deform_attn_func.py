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

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import MultiScaleDeformableAttention as MSDA


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    '''多尺度可变形注意力，根据采样点的位置在多尺度value中插值采样出对对应的特征图，最后和注意力权重进行weighted sum得到输出。'''
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    # 分割得到各特征层对应的value，是一个list。
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    # 由于以下使用了F.grid_sample()，要求采样位置的坐标是归一化到[-1,1]范围（(-1,-1)代表左上角，(1,1)代表右下角）
    # 因此这里是将[0,1]映射到[-1,1]。
    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    # 下面是基于采样点位置插值出对应的采样特征（value）
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # 根据采样点坐标在value中插值出对应的特征。
        # grid_sample()用法：
        #   value_l_充当被插值采样的特征图，是input，维度需要试4d/5d
        #   sampling_grid_l_代表采样的位置，是grid，最后一维2对应input中的坐标，倒数第2、3维代表采样后输出特征图宽、高
        #   input和grid的第一个维度必须一致，最终输出的通道数与input一致，是不变的。
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)

    # 最后将注意力权重和采样特征进行weighted sum
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    # 进行weighted sum
    # (N_*M_, D_, Lq_, L_, P_) -> (N_*M_, D_, Lq_, L_*P_) * (N_*M_, 1, Lq_, L_*P_) =
    # (N_*M_, D_, Lq_, L_*P) -> (N_, M_*D_, Lq_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)

    # (N_, Lq_, M_*D_, )
    return output.transpose(1, 2).contiguous()