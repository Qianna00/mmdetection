import torch
import torch.nn as nn

from mmdet import ops
from mmdet.core import force_fp32
from mmdet.models.builder import ROI_EXTRACTORS
from .single_level import SingleRoIExtractor


@ROI_EXTRACTORS.register_module()
class BaseSubRoIExtractor(SingleRoIExtractor):
    """Extract RoI features from a single level feature map.

    If there are mulitple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56):
        super(BaseSubRoIExtractor, self).__init__(roi_layer=roi_layer,
                                                  out_channels=out_channels,
                                                  featmap_strides=featmap_strides,
                                                  finest_scale=finest_scale)

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None, for_lr=False):
        num_levels = len(feats)
        print(num_levels)
        if not for_lr:
            out_size = self.roi_layers[2].out_size
            if num_levels == 1:
                roi_feats = feats[0].new_zeros(
                    rois.size(0), self.out_channels, *out_size)
                if len(rois) == 0:
                    return roi_feats
                roi_feats = self.roi_layers[2](feats[0], rois)
                return roi_feats
            else:
                roi_feats_sub = self.roi_layers[0](feats[0], rois)
                roi_feats = self.roi_layers[2](feats[1], rois)
                return roi_feats_sub, roi_feats
        roi_feats_lr_sub = self.roi_layers[1](feats[0], rois)
        roi_feats_lr = self.roi_layers[3](feats[1], rois)
        return roi_feats_lr_sub, roi_feats_lr

