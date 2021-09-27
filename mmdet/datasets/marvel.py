import os.path as osp

import mmcv
import numpy as np
from torch.utils.data import Dataset

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class MarvelDataset(Dataset):
    def __init__(self, ann_file, pipeline, test_mode=False, img_prefix=None, seg_prefix=None, proposal_file=None):
        self.images = []
        self.load_images(ann_file)
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file

    def load_images(self, ann_file):
        fh = open(ann_file, 'r')
        dataset = fh.readlines()
        for d in dataset:
            d = d.split(',')
            file_name = d[-1].split('\n')[0]
            self.images.append(file_name)

    def __getitem__(self, idx):
        img_info = dict(filename=self.images[idx])
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def __len__(self):
        return len(self.images)

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['img_shape'] = (256, 256)

