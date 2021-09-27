import os.path as osp

import mmcv
import numpy as np
from torch.utils.data import Dataset

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class MarvelDataset(Dataset):
    def __init__(self, ann_file, pipeline, test_mode=False):
        self.images = self.load_images(ann_file)
        self.pipeline = Compose(pipeline)

    def load_images(self, ann_file):
        fh = open(ann_file, 'r')
        dataset = fh.readlines()
        for d in dataset:
            d = d.split(',')
            file_name = d[-1].split('\n')[0]
            self.images.append(file_name)

    def __getitem__(self, idx):
        return self.pipeline(self.images[idx])
