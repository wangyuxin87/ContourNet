# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d

from maskrcnn_benchmark import layers

class BOUNDARYRCNNC4Predictor(nn.Module):
    def __init__(self, cfg):
        super(BOUNDARYRCNNC4Predictor, self).__init__()
        dim_reduced = cfg.MODEL.ROI_BOUNDARY_HEAD.CONV_LAYERS[-1]
        self.resol = cfg.MODEL.ROI_BOUNDARY_HEAD.RESOLUTION  # 56

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            num_inputs = dim_reduced
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS  #256
            num_inputs = res2_out_channels * stage2_relative_factor

        self.bo_input_xy = Conv2d(num_inputs, num_inputs, 1, 1, 0)
        nn.init.kaiming_normal_(self.bo_input_xy.weight,
                mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bo_input_xy.bias, 0)

        self.conv5_bo_xy = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        nn.init.kaiming_normal_(self.conv5_bo_xy.weight,
                mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv5_bo_xy.bias, 0)

        self.bo_input_1_1 = Conv2d(dim_reduced, dim_reduced, 1, 1, 0)
        nn.init.kaiming_normal_(self.bo_input_1_1.weight,
                                mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bo_input_1_1.bias, 0)

        self.bo_input_2_1 = Conv2d(dim_reduced, dim_reduced, 1, 1, 0)
        nn.init.kaiming_normal_(self.bo_input_2_1.weight,
                                mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bo_input_2_1.bias, 0)

        self.conv5_bo_x = Conv2d(dim_reduced, 1, (3, 1), 1, (1,0)) # H W
        nn.init.kaiming_normal_(self.conv5_bo_x.weight,
                mode='fan_out', nonlinearity='relu') # 'relu'
        nn.init.constant_(self.conv5_bo_x.bias, 0)

        self.conv5_bo_y = Conv2d(dim_reduced, 1, (1, 3), 1, (0,1)) # H W
        nn.init.kaiming_normal_(self.conv5_bo_y.weight,
                mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv5_bo_y.bias, 0)
        self.up_scale=2


    def forward(self, ft):
        ft = self.bo_input_xy(ft)
        ft_2x = self.conv5_bo_xy(ft)

        ft_2x = layers.interpolate(ft_2x, size = (48,48), mode='bilinear', align_corners=True)

        x = self.bo_input_1_1(ft_2x)
        y = self.bo_input_2_1(ft_2x)

        x = self.conv5_bo_x(x)
        y = self.conv5_bo_y(y)

        return x, y



_ROI_KE_PREDICTOR = {"BoundaryRCNNC4Predictor": BOUNDARYRCNNC4Predictor}


def make_roi_boundary_predictor(cfg):
    func = _ROI_KE_PREDICTOR[cfg.MODEL.ROI_BOUNDARY_HEAD.PREDICTOR]
    return func(cfg)
