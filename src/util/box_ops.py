# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
import numpy as np
import math
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    # print('boxes1 shape:', boxes1.shape)
    # print('boxes2 shape:', boxes2.shape)
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def complete_box_iou(boxes1, boxes2):
    """
    Complete IoU

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    b1_xy = boxes1[:, None, :2]
    b1_wh = boxes1[:, None, 2:]
    b1_wh_half = b1_wh / 2
    b1_mins = b1_xy - b1_wh_half
    b1_maxs = b1_xy + b1_wh_half

    b2_xy = boxes2[:, :2]
    b2_wh = boxes2[:, 2:]
    b2_wh_half = b2_wh / 2
    b2_mins = b2_xy - b2_wh_half
    b2_maxs = b2_xy + b2_wh_half

    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxs = torch.min(b1_maxs, b2_maxs)
    intersect_wh = torch.max(intersect_maxs - intersect_mins, torch.zeros_like(intersect_maxs))
    # intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    # b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou, union = box_iou(boxes1, boxes2)
    # print('iou2:', iou)

    # 计算中心的差距
    center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)
    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxs = torch.max(b1_maxs, b2_maxs)
    enclose_wh = torch.max(enclose_maxs - enclose_mins, torch.zeros_like(intersect_maxs))

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    enclose_wh = torch.max(rb - lt, torch.zeros_like(intersect_maxs))

    # 计算对角线距离
    enclose_diagomal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
    ciou = iou - 1.0 * center_distance / (enclose_diagomal + 1e-7)
    v = (4 / (math.pi ** 2)) * torch.pow(
        torch.atan(b1_wh[..., 0] / b1_wh[..., 1]) - torch.atan(b2_wh[..., 0] / b2_wh[..., 1]), 2)

    alpha = v / ((1 - iou) + v)
    ciou = ciou - alpha * v
    ciou = torch.clamp(ciou, min=-1.0, max=1.0)
    return ciou


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def IoU(box1, box2):
    # 计算中间矩形的宽高
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])

    # 计算交集、并集面积
    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    # 计算IoU
    iou = inter / union
    return iou


def IoU_circle(box1, box2):
    # 计算中间矩形的宽高
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])

    # 计算交集、并集面积
    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    # 计算IoU
    iou = inter / union
    return iou


# def DIoU(box1, box2):
#     # 计算对角线长度
#     y1, x1, y2, x2 = box1
#     y3, x3, y4, x4 = box2
#
#     C = np.sqrt((max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) ** 2 + \
#                 (max(y1, y2, y3, y4) - min(y1, y2, y3, y4)) ** 2)
#     # print('C=',C)
#
#     # 计算中心点间距
#     point_1 = ((x2 + x1) / 2, (y2 + y1) / 2)
#     point_2 = ((x4 + x3) / 2, (y4 + y3) / 2)
#     # print('point_1:', point_1)
#     # print('point_2:', point_2)
#
#     D = np.sqrt((point_2[0] - point_1[0]) ** 2 + \
#                 (point_2[1] - point_1[1]) ** 2)
#     # print('D=', D)
#
#     # 计算IoU
#     iou = IoU(box1, box2)
#
#     # 计算空白部分占比
#     lens = D ** 2 / C ** 2
#     diou = iou - lens
#     return diou


if __name__ == "__main__":
    boxes1 = [[0.0, 0, 8, 6], [0.0, 0, 8, 6], [2,4,3,5]]
    boxes2 = [[2.0, 3, 10, 9], [2.0, 3, 10, 9]]
    # boxes1 = torch.Tensor(boxes1)
    # boxes2 = torch.Tensor(boxes2)
    boxes1 = torch.tensor([item for item in boxes1]).cuda()
    boxes2 = torch.tensor([item for item in boxes2]).cuda()

    ciou = complete_box_iou(boxes1, boxes2)
    print(ciou)
