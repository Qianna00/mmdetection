from .coco import CocoDataset
from .builder import DATASETS

@DATASETS.register_module()
class SmdDataset(CocoDataset):
    CLASSES = ('Ferry', 'Buoy', 'Vessel/ship',
               'Speed boat', 'Boat', 'Kayak',
               'Sail boat', 'Swimming person',
               'Flying bird/plane', 'Other')