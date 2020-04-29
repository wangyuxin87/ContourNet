# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import torch
import pandas as pd
from maskrcnn_benchmark.data.datasets.evaluation.word import io_
class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        TO_REMOVE = 1  # TODO remove
        ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
        ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets

    def encode_iou(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        TO_REMOVE = 1  # TODO remove
        ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
        ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets


    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes   # predict  [2, 12000, 4]
            boxes (Tensor): reference boxes.   # anchor  [2, 12000, 4]  xmin0 ymin1 xmax2 ymax3
        """
        boxes = boxes.to(rel_codes.dtype)


        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        ##############################

        pred_boxes = torch.zeros_like(rel_codes)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

        return pred_boxes


    def decode_iou(self, rel_codes, boxes, num_p = 8):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes   # predict  [2, 12000, 4]
            boxes (Tensor): reference boxes.   # anchor  [2, 12000, 4]  xmin0 ymin1 xmax2 ymax3
        """
        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE

        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        # 123
        # 8#4
        # 765
        if num_p == 8:  # 8 boundary points
            x_1 = boxes[:, 0] + widths * rel_codes[:, 0]
            y_1 = boxes[:, 1] + heights * rel_codes[:, 1]
            x_2 = ctr_x + widths * rel_codes[:, 2]
            y_2 = boxes[:, 1] + heights * rel_codes[:, 3]
            x_3 = boxes[:, 2] + widths * rel_codes[:, 4]
            y_3 = boxes[:, 1] + heights * rel_codes[:, 5]
            x_4 = boxes[:, 2] + widths * rel_codes[:, 6]
            y_4 = ctr_y + heights * rel_codes[:, 7]
            x_5 = boxes[:, 2] + widths * rel_codes[:, 8]
            y_5 = boxes[:, 3] + heights * rel_codes[:, 9]
            x_6 = ctr_x + widths * rel_codes[:, 10]
            y_6 = boxes[:, 3] + heights * rel_codes[:, 11]
            x_7 = boxes[:, 0] + widths * rel_codes[:, 12]
            y_7 = boxes[:, 3] + heights * rel_codes[:, 13]
            x_8 = boxes[:, 0] + widths * rel_codes[:, 14]
            y_8 = ctr_y + heights * rel_codes[:, 15]
            x_total = torch.stack([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8], 0)
            y_total = torch.stack([y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8], 0)

        x_min = torch.min(x_total, 0, keepdim=True)  # [1, N]
        x_max = torch.max(x_total, 0, keepdim=True)

        y_min = torch.min(y_total, 0, keepdim=True)
        y_max = torch.max(y_total, 0, keepdim=True)

        N1, N2 = x_min[0].shape
        x_min = x_min[0].view([N2])
        x_max = x_max[0].view([N2])
        y_min = y_min[0].view([N2])
        y_max = y_max[0].view([N2])

        x_min = torch.stack([x_min, ctr_x], 0)
        x_max = torch.stack([x_max, ctr_x], 0)
        y_min = torch.stack([y_min, ctr_y], 0)
        y_max = torch.stack([y_max, ctr_y], 0)

        x_min = torch.min(x_min, 0, keepdim=True)  # [1, N]
        x_max = torch.max(x_max, 0, keepdim=True)
        y_min = torch.min(y_min, 0, keepdim=True)
        y_max = torch.max(y_max, 0, keepdim=True)

        pred_boxes = torch.zeros_like(boxes)

        pred_boxes[:, 0] = x_min[0][0, :]
        pred_boxes[:, 1] = y_min[0][0, :]
        pred_boxes[:, 2] = x_max[0][0, :]
        pred_boxes[:, 3] = y_max[0][0, :]


        return pred_boxes
