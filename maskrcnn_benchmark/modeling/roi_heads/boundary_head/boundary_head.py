# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList

from .roi_boundary_feature_extractors import make_roi_boundary_feature_extractor
from .roi_boundary_predictors import make_roi_boundary_predictor
from .inference import make_roi_boundary_post_processor
from .loss import make_roi_boundary_loss_evaluator

def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


class ROIBOHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIBOHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_boundary_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_boundary_predictor(cfg)
        self.post_processor = make_roi_boundary_post_processor(cfg)
        self.loss_evaluator = make_roi_boundary_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # during training, only focus on positive boxes
            with torch.no_grad():
                # proposals = self.loss_evaluator.subsample(proposals, targets)
                all_proposals = proposals
                proposals, positive_inds = keep_only_positive_boxes(proposals)

        x = self.feature_extractor(features, proposals)
        outputs_x, outputs_y= self.predictor(x)

        if not self.training:
            result = self.post_processor(outputs_x, outputs_y, proposals)

            return x, result, {}, {}, {}

        loss_bo, loss_x, loss_y = self.loss_evaluator(proposals, outputs_x, outputs_y, targets)

        return x, proposals, dict(loss_bo=loss_bo), dict(loss_bo_x=loss_x), dict(loss_bo_y=loss_y)


def build_roi_boundary_head(cfg, in_channels):
    return ROIBOHead(cfg, in_channels)
