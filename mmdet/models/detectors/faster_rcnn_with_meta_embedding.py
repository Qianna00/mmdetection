from ..builder import DETECTORS
from .two_stage_with_MetaEmbedding import TwoStageDetectorMetaEmbedding


@DETECTORS.register_module()
class FasterRCNNMetaEmbedding(TwoStageDetectorMetaEmbedding):

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 init_centroids,
                 neck=None,
                 pretrained=None):
        super(FasterRCNNMetaEmbedding, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_centroids=init_centroids,
            pretrained=pretrained)