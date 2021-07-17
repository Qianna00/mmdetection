from ..builder import DETECTORS
from .two_stage_with_UnsupEmbedding import TwoStageDetectorUnsupEmbedding


@DETECTORS.register_module()
class FasterRCNNUnsupEmbedding(TwoStageDetectorUnsupEmbedding):

    def __init__(self,
                 backbone,
                 backbone_unsup,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 init_centroids,
                 neck=None,
                 pretrained=None,
                 unsup_pretrained=None):
        super(FasterRCNNUnsupEmbedding, self).__init__(
            backbone=backbone,
            backbone_unsup=backbone_unsup,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_centroids=init_centroids,
            pretrained=pretrained,
            unsup_pretrained=unsup_pretrained)
