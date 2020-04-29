import torch
import numpy as np


def iou_regress(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """


    if len(input)==0:
        return input.sum() * 0

    width_i = input[:, 2] - input[:, 0]
    height_i = input[:, 3] - input[:, 1]
    width_t = target[:, 2] - target[:, 0]
    height_t = target[:, 3] - target[:, 1]

    wh_if = torch.zeros_like(width_i)
    wh_if[width_i > 0] += 1
    wh_if[height_i > 0] += 1

    uion_i = width_i * height_i
    uion_t = width_t * height_t

    x_1_max = torch.stack([input[:,0],target[:, 0]], 0)
    y_1_max = torch.stack([input[:,1],target[:, 1]], 0)
    x_2_min = torch.stack([input[:, 2], target[:, 2]], 0)
    y_2_min = torch.stack([input[:, 3], target[:, 3]], 0)

    x_1_max = torch.max(x_1_max, 0, keepdim=True)
    y_1_max = torch.max(y_1_max, 0, keepdim=True)
    x_2_min = torch.min(x_2_min, 0, keepdim=True)
    y_2_min = torch.min(y_2_min, 0, keepdim=True)

    width_inter = x_2_min[0] - x_1_max[0]
    height_inter = y_2_min[0] - y_1_max[0]
    N1, N2 = height_inter.shape
    width_inter = width_inter.view([N2])

    height_inter = height_inter.view([N2])

    inter_area = width_inter * height_inter
    area_union = uion_i + uion_t - inter_area

    wh_if[width_inter > 0] += 1
    wh_if[height_inter > 0] += 1
    wh_if [wh_if != 4] = 0
    wh_if [wh_if > 1] = 1

    inter_area *= wh_if
    area_union *= wh_if

    iou_loss_map = -torch.log((inter_area + 1.0) / (area_union + 1.0))
    iou_loss_map = iou_loss_map * wh_if

    del wh_if
    return iou_loss_map.sum()