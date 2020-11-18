import torch
import torch.nn as nn
import numpy as np

# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from tqdm import tqdm
from mmdet.datasets import build_dataloader, build_dataset
from mmcv import Config
from mmdet.core import bbox2roi
from functools import partial
from torch.utils.data.dataloader import DataLoader


@DETECTORS.register_module()
class TwoStageDetectorMetaEmbedding(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_centroids=False,
                 pretrained=None):
        super(TwoStageDetectorMetaEmbedding, self).__init__()
        self.backbone = build_backbone(backbone)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.init_centroids = init_centroids

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        if self.init_centroids:
            self.centroids = self.roi_head.loss_feat.centroids.data
        else:
            self.centroids = None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)
        if roi_head["type"] == "MetaEmbedding_RoIHead":
            # calculate init_centroids using training dataset
            if init_centroids:
                cfg = Config.fromfile(
                    "/root/data/zq/smd_det/meta_embedding/10c/stage2/faster_rcnn_r50_c4_meta_smd_stage2.py")
                dataset = [build_dataset(cfg.data.centroids_cal)]
                data = build_dataloader(dataset, samples_per_gpu=2, workers_per_gpu=2, num_gpus=2, shuffle=False)
                self.roi_head.loss_feat.centroids.data = self.centroids_cal(data)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetectorMetaEmbedding, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                    x,
                    img_metas,
                    gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        """roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)"""


        roi_losses = self.roi_head(x,
                                   centroids=self.centroids,
                                   img_metas=img_metas,
                                   proposal_list=proposal_list,
                                   gt_bboxes=gt_bboxes,
                                   gt_labels=gt_labels,
                                   gt_bboxes_ignore=gt_bboxes_ignore,
                                   gt_masks=gt_masks,
                                   test=False,
                                   **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        # assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head(x,
                             centroids=self.centroids,
                             proposal_list=proposal_list,
                             img_metas=img_metas,
                             test=True)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def centroids_cal(self, data):

        centroids = torch.zeros(self.roi_head.num_classes,
                                self.roi_head.feat_dim).cuda()

        print('Calculating centroids.')

        # Calculate initial centroids only on training data.
        with torch.set_grad_enabled(False):

            for inputs in tqdm(data):
                imgs, gt_labels, gt_bboxes, img_metas = inputs["img"], \
                                                        inputs["gt_labels"], \
                                                        inputs["gt_bboxes"],\
                                                        inputs["img_metas"]
                # Calculate Features of each training data
                feats = self.backbone(imgs)
                proposal_list = self.rpn_head.simple_test_rpn(feats, img_metas)
                num_imgs = len(img_metas)
                # if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
                sampling_results = []
                for i in range(num_imgs):
                    assign_result = self.roi_head.std_roi_head.bbox_assigner.assign(
                        proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                        gt_labels[i])
                    sampling_result = self.roi_head.std_roi_head.bbox_sampler.sample(
                        assign_result,
                        proposal_list[i],
                        gt_bboxes[i],
                        gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in feats])
                    sampling_results.append(sampling_result)

                rois = bbox2roi([res.bboxes for res in sampling_results])
                bbox_feats = self.roi_head.std_roi_head.bbox_roi_extractor(
                    feats[:self.roi_head.std_roi_head.bbox_roi_extractor.num_inputs], rois)

                labels = self.roi_head.std_roi_head.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                                                gt_labels, self.train_cfg)[0]

                # Add all calculated features to center tensor
                for i in range(len(labels)):
                    label = labels[i]
                    centroids[label] += bbox_feats[i]

        # Average summed features with class count
        centroids /= torch.tensor(class_count(data)).float().unsqueeze(1).cuda()

        return centroids


def class_count(data):
    labels = np.array(data.dataset.labels)
    class_data_num = []
    for l in np.unique(labels):
        class_data_num.append(len(labels[labels == l]))
    return class_data_num