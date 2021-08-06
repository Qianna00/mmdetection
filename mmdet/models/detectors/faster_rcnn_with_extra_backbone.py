from ..builder import DETECTORS
from .two_stage_extra_backbone import TwoStageDetectorWithExtraBackbone


@DETECTORS.register_module()
class FasterRCNNWithExtraBackbone(TwoStageDetectorWithExtraBackbone):

    def __init__(self,
                 backbone,
                 extra_backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 pretrained_extra=None):
        super(FasterRCNNWithExtraBackbone, self).__init__(
            backbone=backbone,
            extra_backbone=extra_backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            pretrained_extra=pretrained_extra)
