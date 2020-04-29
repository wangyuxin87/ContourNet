# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
import math
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.rpn.retinanet.retinanet import build_retinanet
from maskrcnn_benchmark.modeling.rpn.fcos.fcos import build_fcos
from .loss import make_rpn_loss_evaluator
from .anchor_generator import make_anchor_generator
from .inference import make_rpn_postprocessor


class RPNHeadConvRegressor(nn.Module):
    """
    A simple RPN Head for classification and bbox regression
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHeadConvRegressor, self).__init__()
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        logits = [self.cls_logits(y) for y in x]
        bbox_reg = [self.bbox_pred(y) for y in x]

        return logits, bbox_reg


class RPNHeadFeatureSingleConv(nn.Module):
    """
    Adds a simple RPN Head with one conv to extract the feature
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
        """
        super(RPNHeadFeatureSingleConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        for l in [self.conv]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

        self.out_channels = in_channels

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        x = [F.relu(self.conv(z)) for z in x]

        return x


@registry.RPN_HEADS.register("SingleConvRPNHead_1")
class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred_new = nn.Conv2d(
            in_channels, num_anchors * 18, kernel_size=1, stride=1
        )

        for l in [self.conv, self.cls_logits, self.bbox_pred_new]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):

        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred_new(t))
        return logits, bbox_reg


class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg, in_channels):
        super(RPNModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator(cfg)

        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
        head = rpn_head(
            cfg, in_channels, anchor_generator.num_anchors_per_location()[0]
        )

        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True)
        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)

        loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None, prefix=''):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        objectness, rpn_box_regression = self.head(features)   # len = 5
        anchors = self.anchor_generator(images, features)

        if self.training:
            return self._forward_train(anchors, objectness,
                                       rpn_box_regression, targets, prefix)
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)

    def _forward_train(self, anchors, objectness, rpn_box_regression,  # [image,number,[n,4]]
                       targets, prefix):
        if self.cfg.MODEL.RPN_ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            boxes = anchors
        else:
            # print('\n---end-to-end model---\n')
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch.
            with torch.no_grad():
                boxes = self.box_selector_train(
                    anchors, objectness, rpn_box_regression, targets
                )
        anchors_new = list(zip(*anchors))
        regress_new = regress_to_box(anchors_new, rpn_box_regression)

        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
            anchors, objectness, regress_new, targets
        )
        losses = {
            prefix + "loss_objectness": loss_objectness,
            prefix + "loss_rpn_box_reg": loss_rpn_box_reg,
        }
        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes, {}


def build_rpn(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    if cfg.MODEL.FCOS_ON:
        return build_fcos(cfg, in_channels)
    if cfg.MODEL.RETINANET_ON:
        return build_retinanet(cfg, in_channels)

    return RPNModule(cfg, in_channels)


def regress_to_box(anchor_define,regress_pre):

    boxes_total = []
    num_f = 0
    for a, b in zip(anchor_define, regress_pre):
        boxes_total.append(forward_feature_map(a, b))
        num_f += 1
    return boxes_total

def forward_feature_map(anchors_define, boxes_regression):
    N, A, H, W = boxes_regression.shape

    boxes_regression = faltten(boxes_regression, N, A, 18, H, W)  #

    # image_shapes = [box.size for box in anchors_define]
    concat_anchors = torch.cat([a.bbox for a in anchors_define], dim=0)
    concat_anchors = concat_anchors.reshape(N, -1, 4)
    proposals = decode_iou(boxes_regression.view(-1, 18), concat_anchors.view(-1, 4))
    box_temp_post = proposals.view(N, -1, 4)

    return box_temp_post

def faltten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)  #N H W A C
    layer = layer.reshape(N, -1, C)  # N H*W*A C
    return layer

def decode_iou( rel_codes, boxes, num_p = 8):
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
        dx = rel_codes[:, 16]
        dy = rel_codes[:, 17]

        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        ctr_x_new = dx * widths * 0.5 + ctr_x
        ctr_y_new = dy * heights * 0.5 + ctr_y
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
            x_total = torch.stack([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8], 0)  # [8, N]
            y_total = torch.stack([y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8], 0)

        x_min = torch.min(x_total, 0, keepdim=True)  # [1, N]
        x_max = torch.max(x_total, 0, keepdim=True)  # [1, N]
        y_min = torch.min(y_total, 0, keepdim=True)  # [1, N]
        y_max = torch.max(y_total, 0, keepdim=True)  # [1, N]

        N1, N2 = x_min[0].shape
        x_min = x_min[0].view([N2])
        x_max = x_max[0].view([N2])
        y_min = y_min[0].view([N2])
        y_max = y_max[0].view([N2])

        x_min = torch.stack([x_min, ctr_x_new], 0)
        x_max = torch.stack([x_max, ctr_x_new], 0)
        y_min = torch.stack([y_min, ctr_y_new], 0)
        y_max = torch.stack([y_max, ctr_y_new], 0)

        x_min = torch.min(x_min, 0, keepdim=True)  # [1, N]
        x_max = torch.max(x_max, 0, keepdim=True)  # [1, N]
        y_min = torch.min(y_min, 0, keepdim=True)  # [1, N]
        y_max = torch.max(y_max, 0, keepdim=True)  # [1, N]

        pred_boxes = torch.zeros_like(boxes)

        pred_boxes[:, 0] = x_min[0][0, :]
        pred_boxes[:, 1] = y_min[0][0, :]
        pred_boxes[:, 2] = x_max[0][0, :]
        pred_boxes[:, 3] = y_max[0][0, :]

        return pred_boxes