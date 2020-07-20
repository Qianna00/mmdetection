from ..builder import DETECTORS
from .two_stage_with_gan import TwoStageGanDetector

@DETECTORS.register_module()
class FasterRCNNGan(TwoStageGanDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(FasterRCNNGan, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
