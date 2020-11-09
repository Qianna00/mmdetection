import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .standard_roi_head import StandardRoIHead
from ..builder import HEADS, build_head, build_loss
from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger



@HEADS.register_module()
class MetaEmbedding_RoIHead(nn.Module):
    def __init__(self,
                 num_classes=7,
                 feat_dim=2048,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 loss_feat=dict(
                     type="DiscCentroidsLoss",
                     num_classes=7,
                     feat_dim=2048,
                     size_average=True
                 )
                 ):
        super(MetaEmbedding_RoIHead, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.fc_hallucinator = nn.Linear(feat_dim, num_classes)
        self.fc_selector = nn.Linear(feat_dim, feat_dim)
        self.std_roi_head = StandardRoIHead(bbox_roi_extractor=bbox_roi_extractor,
                                            bbox_head=bbox_head,
                                            shared_head=shared_head,
                                            train_cfg=train_cfg,
                                            test_cfg=test_cfg)
        if loss_feat is not None:
            self.loss_feat = build_loss(loss_feat)

    def forward(self,
                x,
                centroids=None,
                img_metas=None,
                proposal_list=None,
                gt_bboxes=None,
                gt_labels=None,
                gt_bboxes_ignore=None,
                gt_masks=None,
                test=False,
                *args):

        if centroids is not None:

            # storing direct feature
            direct_feature = x.clone()

            # batch_size = x.size(0)
            # feat_size = x.size(1)

            # set up visual memory
            # x_expand = x.clone().unsqueeze(1).expand(-1, self.num_classes, -1)
            # centroids_expand = centroids.clone().unsqueeze(0).expand(batch_size, -1, -1)
            keys_memory = centroids.clone()

            # computing memory feature by querying and associating visual memory
            values_memory = self.fc_hallucinator(x.clone())
            values_memory = values_memory.softmax(dim=1)
            memory_feature = torch.matmul(values_memory, keys_memory)

            # computing concept selector
            concept_selector = self.fc_selector(x.clone())
            concept_selector = concept_selector.tanh()
            x = direct_feature + concept_selector * memory_feature

            # storing infused feature
            # infused_feature = concept_selector * memory_feature

        if not test:
            roi_losses = self.std_roi_head.forward_train(x,
                                                         img_metas,
                                                         proposal_list,
                                                         gt_bboxes,
                                                         gt_labels,
                                                         gt_bboxes_ignore,
                                                         gt_masks)
            if centroids is not None:
                feat_loss = self.loss_feat(x, gt_labels)
                roi_losses.update(loss_feat=feat_loss)
                # roi_losses.update(features=[direct_feature, infused_feature])
            return roi_losses
        else:
            bbox_results = self.std_roi_head.simple_test(x,
                                                         proposal_list,
                                                         img_metas,
                                                         proposals=None,
                                                         rescale=False)
            return bbox_results

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            nn.init.normal_(self.fc_hallucinator.weight, 0, 0.01)
            nn.init.constant_(self.fc_hallucinator.bias, 0)
            nn.init.normal_(self.fc_selector.weight, 0, 0.001)
            nn.init.constant_(self.fc_selector.bias, 0)
            self.std_roi_head.init_weights(pretrained)
        else:
            raise TypeError('pretrained must be a str or None')

