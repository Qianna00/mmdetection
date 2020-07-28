import torch
import numpy as np
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, bbox_mapping, \
    merge_aug_bboxes, multiclass_nms
from ..builder import HEADS, build_head, build_roi_extractor, build_neck
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin


@HEADS.register_module()
class RoIHeadGan(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head.
    """
    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 fsr_generator=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 dis_head=None,
                 train_cfg=None,
                 test_cfg=None
                 ):
        BaseRoIHead.__init__(self,
                             bbox_roi_extractor=bbox_roi_extractor,
                             bbox_head=bbox_head,
                             mask_roi_extractor=mask_roi_extractor,
                             mask_head=mask_head,
                             shared_head=shared_head,
                             train_cfg=train_cfg,
                             test_cfg=test_cfg)

        if fsr_generator is not None:
            self.init_fsr_generator(fsr_generator)

        if dis_head is not None:
            self.init_dis_head(dis_head)

    @property
    def with_fsr_generator(self):
        return hasattr(self, 'fsr_generator') and self.fsr_generator is not None

    @property
    def with_dis_head(self):
        return hasattr(self, 'fsr_generator') and self.fsr_generator is not None

    def init_assigner_sampler(self):
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def init_fsr_generator(self, fsr_generator):
        self.fsr_generator = build_neck(fsr_generator)

    def init_dis_head(self, dis_head):
        self.dis_head = build_head(dis_head)

    def init_weights(self, pretrained):
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights(pretrained)
            if self.with_fsr_generator:
                self.fsr_generator.init_weights(pretrained)
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()
        if self.with_dis_head:
            self.dis_head.init_weights(pretrained)


    def forward_dummy(self, x, proposals):
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      x_lr=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposals (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # feats = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas, x_lr)
            losses.update(bbox_results['loss_bbox'])
            """if self.with_dis_head:
                feats.update(bbox_feats=bbox_results['bbox_feats'])
                feats.update(bbox_feats_lr=bbox_results['bbox_feats_lr'])
                feats.update(num_rois_hr=bbox_results['num_rois_hr'])
                feats.update(num_rois_sr=bbox_results['num_rois_sr'])"""
            if self.with_dis_head:
                losses.update(loss_b=bbox_results['loss_det'])
                losses.update(loss_g=bbox_results['loss_gen'])
                losses.update(loss_d=bbox_results['loss_dis'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            # TODO: Support empty tensor input. #2280
            if mask_results['loss_mask'] is not None:
                losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, rois, rois_index_hr, rois_index_sr, rois_index_small, x_lr=None):
        # TODO: a more flexible way to decide which feature maps to use

        # bbox_feats = self.bbox_roi_extractor(x, rois)
        if self.with_fsr_generator:
            bbox_feats_sub_hr, bbox_feats_hr = self.bbox_roi_extractor(x, rois)
            bbox_feats_hr = self.fsr_generator((bbox_feats_sub_hr, bbox_feats_hr))
            del bbox_feats_sub_hr
            if x_lr is not None:
                bbox_feats_sub_lr, bbox_feats_lr = self.bbox_roi_extractor(x_lr, rois[rois_index_sr], for_lr=True)
                bbox_feats_lr = self.fsr_generator((bbox_feats_sub_lr, bbox_feats_lr))
                del bbox_feats_sub_lr
        """cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)"""
        # bbox_results = dict(bbox_feats=bbox_feats)
        bbox_results = {}
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats_hr)
            # if x_lr is not None:
                # bbox_feats_lr = self.shared_head(bbox_feats_sr[rois_index_small])
        # if x_lr is not None:
        cls_score, bbox_pred = self.bbox_head(bbox_feats[rois_index_small])
        del bbox_feats
        bbox_results.update(cls_score=cls_score)
        bbox_results.update(bbox_pred=bbox_pred)

        if self.with_dis_head:
            dis_score_hr = self.dis_head(bbox_feats_hr[rois_index_hr])
            bbox_results.update(dis_score_hr=dis_score_hr)
            if x_lr is not None:
                dis_score_sr = self.dis_head(bbox_feats_lr)
                bbox_results.update(dis_score_sr=dis_score_sr)
            """bbox_results.update(bbox_feats=bbox_feats)
            bbox_results.update(bbox_feats_lr=bbox_feats_sr)"""

        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, x_lr):
        rois = bbox2roi([res.bboxes for res in sampling_results])
        areas = torch.mul((rois[:, 3] - rois[:, 1]), rois[:, 4] - rois[:, 2])
        rois_index_hr = torch.where(areas > 96 * 96)
        rois_index_sr = torch.where(areas <= 96 * 96 * 4)
        rois_index_small = torch.where(areas <= 96 * 96)

        bbox_results = self._bbox_forward(x, rois, rois_index_hr, rois_index_sr, rois_index_small, x_lr)

        bbox_targets = self.bbox_head.get_targets(sampling_results[rois_index_small], gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois[rois_index_small],
                                        *bbox_targets)
        bbox_results.update(loss_bbox=loss_bbox)
        # if x_lr is not None:

        """bbox_targets_lr = bbox_targets[0][rois_index_small], bbox_targets[1][rois_index_small], \
                              bbox_targets[2][rois_index_small], bbox_targets[3][rois_index_small]

            loss_bbox_lr = self.bbox_head.loss(bbox_results['cls_score_lr'][rois_index_small],
                                               bbox_results['bbox_pred_lr'][rois_index_small],
                                               rois[rois_index_small],
                                               *bbox_targets_lr)"""

        """bbox_results.update(num_rois_hr=rois_index_hr[0].shape[0])
            bbox_results.update(num_rois_sr=rois_index_sr[0].shape[0])
            bbox_results.update(bbox_feats=bbox_results['bbox_feats'][rois_index_hr])
            bbox_results.update(bbox_feats_lr=bbox_results['bbox_feats_lr'][rois_index_sr])"""

        target_ones_g = torch.Tensor(np.ones((rois_index_sr[0].shape[0], 1))).cuda().long()
        target_ones_d = torch.Tensor(np.ones((rois_index_hr[0].shape[0], 1))).cuda().long()
        target_zeros_d = torch.Tensor(np.zeros((rois_index_sr[0].shape[0], 1))).cuda().long()
        # dis_score_sr = bbox_results['dis_score_sr'][rois_index_sr]
        # dis_score_hr = bbox_results['dis_score_hr'][rois_index_hr]

        loss_g_dis = self.dis_head.loss(bbox_results['dis_score_sr'], target_ones_g)
        loss_det = loss_bbox['loss_cls'] + loss_bbox['loss_bbox']
        loss_gen = loss_det + loss_g_dis
        loss_dis = (self.dis_head.loss(bbox_results['dis_score_sr'], target_zeros_d) + self.dis_head.loss(
                bbox_results['dis_score_hr'], target_ones_d)) / 2
        bbox_results.update(loss_gen=loss_gen)
        bbox_results.update(loss_dis=loss_dis)
        bbox_results.update(loss_det=loss_det)
        bbox_results.update(loss_bbox=loss_bbox)

        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            if pos_rois.shape[0] == 0:
                return dict(loss_mask=None)
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)
            if pos_inds.shape[0] == 0:
                return dict(loss_mask=None)
            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward_test(x, rois)

        img_shape = img_metas[0]['img_shape']
        scale_factor = img_metas[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_bboxes(
            rois,
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']
            # TODO more flexible
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            rois = bbox2roi([proposals])
            # recompute feature maps to save GPU memory
            bbox_results = self._bbox_forward_test(x, rois)
            bboxes, scores = self.bbox_head.get_bboxes(
                rois,
                bbox_results['cls_score'],
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels

    def _bbox_forward_test(self, x, rois):
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(x[1], rois)
        if self.with_fsr_generator:
            bbox_feats_sub, bbox_feats = self.bbox_roi_extractor(x, rois)
            bbox_feats_sr = self.fsr_generator((bbox_feats_sub, bbox_feats))
            areas = torch.mul((rois[:, 3] - rois[:, 1]), rois[:, 4] - rois[:, 2])
            rois_small_index = torch.where(areas < 96*96)
            bbox_feats[rois_small_index] = bbox_feats_sr[rois_small_index]
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        return bbox_results
