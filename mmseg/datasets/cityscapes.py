# Copyright (c) OpenMMLab. All rights reserved.
from typing import List
import os.path as osp
import re
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CityscapesDataset(BaseSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,
                                                    30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)


@DATASETS.register_module()
class ACDCDataset(CityscapesDataset):
    def __init__(self, img_suffix='_rgb_anon.png',
                 seg_map_suffix='_gt_labelTrainIds.png',
                 **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)


@DATASETS.register_module()
class GTADataset(CityscapesDataset):
    def __init__(self, img_suffix='.png',
                 seg_map_suffix='_labelTrainIds.png',
                 **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)


@DATASETS.register_module()
class SynthiaDataset(CityscapesDataset):
    def __init__(self, img_suffix='.png',
                 seg_map_suffix='_labelTrainIds.png',
                 **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)


@DATASETS.register_module()
class CityscapesDataset_foggy(CityscapesDataset):
    def __init__(self, **kwargs):
        self.train_txt = kwargs.pop('train_txt', None)
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        with open(self.train_txt) as f:
            all_files = f.readlines()
        for _, line in enumerate(all_files):
            img_name = line.strip()
            data_info = dict(img_path=osp.join(img_dir, img_name))
            if ann_dir is not None:
                seg_map = re.findall(re.compile(r'(.+)_foggy'), img_name)[0] + self.seg_map_suffix
                seg_map = re.sub('leftImg8bit_', '', seg_map)  # gt do not have leftImg8bit
                # seg_map = img_name + seg_map_suffix
                data_info['seg_map_path'] = osp.join(ann_dir, seg_map)  # 这里也得修改
            data_info['label_map'] = self.label_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            data_list.append(data_info)
        print('Load {} data to train!'.format(len(data_list)))
        return data_list


@DATASETS.register_module()
class CityscapesDataset_rainy(CityscapesDataset):
    def __init__(self, **kwargs):
        self.train_txt = kwargs.pop('train_txt', None)
        super().__init__(img_suffix='_depth_rain.png',
                         seg_map_suffix='_gtFine_labelTrainIds.png',
                         **kwargs)

    def load_data_list(self) -> List[dict]:
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        with open(self.train_txt) as f:
            all_files = f.readlines()
        for _, line in enumerate(all_files):
            img_name = line.strip()
            data_info = dict(img_path=osp.join(img_dir, img_name))
            if ann_dir is not None:
                seg_map = re.findall(re.compile(r'(.+)_rain'), img_name)[0] + self.seg_map_suffix
                seg_map = re.sub('leftImg8bit_', '', seg_map)  # gt do not have leftImg8bit
                # seg_map = img_name + seg_map_suffix
                data_info['seg_map_path'] = osp.join(ann_dir, seg_map)  # 这里也得修改
            data_info['label_map'] = self.label_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            data_list.append(data_info)
        print('Load {} data to train!'.format(len(data_list)))
        return data_list