from typing import List
import os.path as osp
import re
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class NTHUDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])
    # NTHU 这里只有13个share classes，但是因为是source model训练了19个class，如果直接映射到13个class会有问题。所以做法和Synthia一样
    # 同样是16个classes，只不过最后算的时候不算那6个class的结果
    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_eval.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
        self.ignore_label = 255
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        classes_13 = [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
        self.label_map = {c_id: i for i, c_id in enumerate(classes_13)}
        # self.full_init()
        self.data_list = self.load_data_list()
