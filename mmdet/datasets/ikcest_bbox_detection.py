from .coco import CocoDataset
from .builder import DATASETS

@DATASETS.register_module()
class IKCESTDetDataset(CocoDataset):
    CLASSES = ('Motor Vehicle', 'Non-motorized Vehicle',
               'Pedestrian', 'Red Light', 'Yellow Light',
               'Green Light', 'Light off')