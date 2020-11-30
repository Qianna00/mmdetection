import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .standard_roi_head import StandardRoIHead
from ..builder import HEADS, build_head, build_loss
from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger
from mmdet.core import bbox2result, bbox2roi



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
        self.pool_meta_embedding = nn.AvgPool2d((14, 14))
        self.fc_hallucinator = nn.Linear(self.feat_dim, self.num_classes)
        self.fc_selector = nn.Linear(self.feat_dim, self.feat_dim)
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
        if not test:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.std_roi_head.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.std_roi_head.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_targets = self.std_roi_head.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                                   gt_labels, self.std_roi_head.train_cfg)
        else:
            print("proposal_list:", proposal_list)
            rois = bbox2roi(proposal_list)
            print("rois:", rois)
        bbox_feats = self.std_roi_head.bbox_roi_extractor(
            x[:self.std_roi_head.bbox_roi_extractor.num_inputs], rois)

        if centroids is not None:
            if not test:
                pos_index = torch.nonzero(bbox_targets[0]-10).squeeze(1)
                # print("pos_index:", torch.nonzero(bbox_targets[0]-10).size())
                bbox_feats_pos = bbox_feats[pos_index]
                bbox_feats_pos = self.get_meta_embedding_feature(bbox_feats_pos, centroids)
                bbox_feats[pos_index] = bbox_feats_pos
                feat_loss = self.loss_feat(bbox_feats_pos, bbox_targets[0][pos_index])
            else:
                bbox_feats = self.get_meta_embedding_feature(bbox_feats, centroids)

        if self.std_roi_head.with_shared_head:
            bbox_feats = self.std_roi_head.shared_head(bbox_feats)

        if not test:
            roi_losses = dict()
            cls_score, bbox_pred = self.std_roi_head.bbox_head(bbox_feats)

            loss_bbox = self.std_roi_head.bbox_head.loss(cls_score,
                                            bbox_pred, rois,
                                            *bbox_targets)
            roi_losses.update(loss_bbox)

            if centroids is not None:
                roi_losses.update(loss_feat=feat_loss)
                # roi_losses.update(features=[direct_feature, infused_feature])
            return roi_losses
        else:
            bbox_results = self.std_roi_head.simple_test(bbox_feats,
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

    def get_meta_embedding_feature(self, feats, centroids):

        # storing direct feature
        direct_feature = feats.clone()

        # batch_size = feats.size(0)
        # feat_size = x.size(1)

        # set up visual memory
        # x_expand = x.clone().unsqueeze(1).expand(-1, self.num_classes, -1)
        # centroids_expand = centroids.clone().unsqueeze(0).expand(batch_size, -1, -1)
        keys_memory = centroids.clone().cuda()

        pooled_feats = self.pool_meta_embedding(feats.clone()).squeeze()
        if len(pooled_feats.size()) != 2:
            pooled_feats = pooled_feats.unsqueeze(0)

        # computing memory feature by querying and associating visual memory
        values_memory = self.fc_hallucinator(pooled_feats)
        # print(pooled_feats.size(), values_memory.size())

        values_memory = values_memory.softmax(dim=1)
        memory_feature = torch.mm(values_memory, keys_memory.view(self.num_classes, -1))

        # computing concept selector
        concept_selector = self.fc_selector(pooled_feats)
        concept_selector = concept_selector.tanh()
        feats = direct_feature + concept_selector.unsqueeze(2).unsqueeze(3).expand(-1, -1, feats.size(2), feats.size(3))\
                * memory_feature.view(feats.size(0), feats.size(1), feats.size(2), feats.size(3))

        # storing infused feature
        # infused_feature = concept_selector * memory_feature
        return feats
