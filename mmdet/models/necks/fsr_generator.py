import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.cnn import constant_init, kaiming_init
from mmdet.core import auto_fp16
from ..builder import NECKS
from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, eps=0.8),
            nn.ReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, eps=0.8)
        )
        self.relu = nn.ReLU()

    def forward(self, inputs):
        return self.relu(inputs + self.conv_block(inputs))


@NECKS.register_module()
class FSRGenerator(nn.Module):
    def __init__(self, in_channels, num_block):
        super(FSRGenerator, self).__init__()

        res_blocks = []

        for i in range(num_block):
            res_blocks.append(ResidualBlock(in_channels))

        self.res_blocks = nn.Sequential(*res_blocks)

    def forward(self, inputs):
        pooled_regions_sub, pooled_regions = inputs
        feat = torch.cat((pooled_regions, pooled_regions_sub), dim=1)  # 将channel所在维度concat

        feat = self.res_blocks(feat)
        feat = feat[:, :1024, :, :]  # 只取 feat_base部分，通道数为1024

        return feat

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.res_blocks:
                for n in m.conv_block:
                    if isinstance(n, nn.Conv2d):
                        kaiming_init(n)
                    elif isinstance(n, nn.BatchNorm2d):
                        constant_init(n, 1)
        else:
            raise TypeError('pretrained must be a str or None')


