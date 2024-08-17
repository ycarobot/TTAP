import torch
import csv
import torch.nn.functional as F
from tqdm import tqdm
from mmengine.runner import autocast
import pycocotools.mask as maskUtils
import copy
import os
import numpy as np
from collections import OrderedDict, Counter
import mmcv
import mmengine
import re
from copy import deepcopy
# for visualization
from torchvision.utils import save_image
from mmseg.visualization import SegLocalVisualizer


def CHECK_NUM_PARAMS(model, lr=0.0):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Len params to train:{} and Lr is {}'.format(params, lr))
    return params


def create_ema_model(model):
    ema_model = deepcopy(model)#get_model(args.model)(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    #_, availble_gpus = self._get_available_devices(self.config['n_gpu'])
    #ema_model = torch.nn.DataParallel(ema_model, device_ids=availble_gpus)
    return ema_model


class SourceOnly:
    def __init__(self, cfg, model, dataloader, work_dir, runner, efficient_test=False, **kwargs):
        # Cityscapes dataset
        self.id2label = {
            "0": "road",
            "1": "sidewalk",
            "2": "building",
            "3": "wall",
            "4": "fence",
            "5": "pole",
            "6": "traffic light",
            "7": "traffic sign",
            "8": "vegetation",
            "9": "terrain",
            "10": "sky",
            "11": "person",
            "12": "rider",
            "13": "car",
            "14": "truck",
            "15": "bus",
            "16": "train",
            "17": "motorcycle",
            "18": "bicycle"
        }
        self.CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
                        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                        'bicycle')  # Cityscapes, ACDC, Cs-foggy, CS-rainy
        self.PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                        [0, 80, 100], [0, 0, 230], [119, 11, 32]]
        self.cfg = cfg
        self.runner = runner
        self.model = model
        self.dataloader = dataloader
        self.evaluator = kwargs['evaluator']
        self.work_dir = work_dir
        self.efficient_test = efficient_test
        self.fp16 = True  # 提高精度减少内存
        self.visualizer = SegLocalVisualizer(vis_backends=[dict(type='LocalVisBackend')], save_dir=work_dir)
        self.visualizer.dataset_meta = dict(
            classes=self.CLASSES,
            palette=self.PALETTE)

    def process_batch_class_label_nthu(self, data):
        if self.dataloader.dataset.label_map is not None:
            # process NTHU dataset
            label_map1 = self.dataloader.dataset.id_to_trainid  # 全部计算
            label_map2 = self.dataloader.dataset.label_map  # 13 classes
            for i in range(len(data['data_samples'])):
                for l1 in label_map1:
                    data['data_samples'][i].gt_sem_seg.data[data['data_samples'][i].gt_sem_seg.data == l1] = \
                        label_map1[l1]
                # unique_label = torch.unique(data['data_samples'][i].gt_sem_seg.data)

                # labels = torch.zeros_like(data['data_samples'][i].gt_sem_seg.data).fill_(255)
                # for k, v in label_map2.items():
                #     labels[data['data_samples'][i].gt_sem_seg.data == k] = v
                # data['data_samples'][i].gt_sem_seg.data = labels
        return data

    def set_model_status(self):
        self.model.eval()

    def save_result(self, predict, filename):
        predict = copy.deepcopy(predict.detach().cpu().squeeze())
        # print(Counter(predict.numpy().flatten()))
        if self.efficient_test:
            semantic_class_in_img = torch.unique(predict)  # 出现了什么类别
            semantic_bitmasks, semantic_class_names = [], []
            anns = {'semantic_mask': {}}
            for i in range(len(semantic_class_in_img)):
                class_name = self.id2label[str(semantic_class_in_img[i].item())]
                class_mask = predict == semantic_class_in_img[i]  # 当前class的位置
                class_mask = class_mask.numpy().astype(np.uint8)
                semantic_class_names.append(class_name)
                semantic_bitmasks.append(class_mask)
                # keys: size, counts
                # RLE编码(run-length encoding)
                anns['semantic_mask'][str(semantic_class_in_img[i].item())] = maskUtils.encode(
                    np.array((predict == semantic_class_in_img[i]).cpu().numpy(), order='F', dtype=np.uint8))
                anns['semantic_mask'][str(semantic_class_in_img[i].item())]['counts'] = \
                    anns['semantic_mask'][str(semantic_class_in_img[i].item())]['counts'].decode('utf-8')

            mmengine.dump(anns, '{}/{}.json'.format(self.work_dir, filename))

    def build_optimizer(self):
        pass

    @torch.no_grad()
    def forward_epoch(self, epoch, data, optimizer=None):
        with autocast(enabled=False):
            # data.keys: inputs, data_samples （里面是gt）.  两个都是list
            # inputs的device是cpu不是cuda，应该再model里面转换了
            # print(data['inputs'][0].shape)
            outputs = self.model.test_step(data)  # bs =1
            # 常用的两个就是pred_sem_seg和seg_logits

        return outputs

    def forward_train(self):
        optimizer = self.build_optimizer()
        tbar = tqdm(self.dataloader)
        # tbar = self.dataloader
        for epoch, data in enumerate(tbar):
            data = self.process_batch_class_label_nthu(data)
            self.set_model_status()
            outputs = self.forward_epoch(epoch, data, optimizer=optimizer)

            if self.efficient_test:
                self.save_result(outputs.pred_sem_seg.data, outputs.img_path.split('/')[-1][:-4])
            else:
                self.evaluator.process(data_samples=outputs, data_batch=data)

            # self.visualizer.add_datasample(
            #     name=os.path.join(self.work_dir, data['data_samples'][0].img_path.split('/')[-1]),
            #     image=mmcv.imread(data['data_samples'][0].img_path),
            #     data_sample=outputs[0],  # 一个bs那样处理
            #     show=False,
            #     draw_gt=False,
            #     with_labels=False,
            # )

        if not self.efficient_test:
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            print(metrics)
            with open(os.path.join(self.work_dir, 'result.csv'), 'a+') as f:
                writer = csv.writer(f)
                writer.writerow([metrics['aAcc'], metrics['mAcc'], metrics['mIoU']])
            torch.save(metrics, os.path.join(self.work_dir, 'result.pkl'))
        else:
            ret_metrics = self.evaluator.metrics[0].ret_metrics
            ret_metrics = OrderedDict({
                ret_metric: np.round(np.nan_to_num(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            })
            agg_IoU = np.append(ret_metrics['IoU'], np.mean(ret_metrics['IoU']))
            agg_IoU = np.round(agg_IoU, 1)
            with open(os.path.join(self.work_dir, 'all_IoU.csv'), 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(agg_IoU)
            with open(os.path.join(self.work_dir, 'all_Acc.csv'), 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(ret_metrics['Acc'])

    def get_gt_path(self, filenames, path=None):
        gt_files = []
        # ACDC
        if 'fog' in path or 'snow' in path or 'night' in path or 'rain' in path:
            domain = path.split('/')[-1]
            gt_path = './data/Segmentation/ACDC/gt/{}/train'.format(domain)
            for f in filenames:
                root_path = gt_path + '/' + f.split('_')[0].split('/')[0]
                # root_path = gt_path
                pattern = re.compile('\\w+_frame_\\d+')
                gt_file = re.findall(pattern, f)[0] + '_gt_labelTrainIds.png'
                gt_file = os.path.join(root_path, gt_file)
                gt_files.append(gt_file)

        elif 'Cityscapes' in path:
            gt_path = './data/Segmentation/Cityscapes/gtFine/train'
            for f in filenames:
                # frankfurt  lindau  munster
                root_path = gt_path + '/' + f.split('_')[0]
                pattern = re.compile('\\w+_\\d+_\\d+')  # 跟新的规则使用两个\\表示，以前是一个
                gt_file = re.findall(pattern, f)[0] + '_gtFine_labelTrainIds.png'
                gt_file = os.path.join(root_path, gt_file)
                gt_files.append(gt_file)

        elif 'gta2cs' in path or 'synthia2cs' in path or 'foggy' in path or 'rainy' in path:
            gt_path = './data/Segmentation/Cityscapes/gtFine/val'
            for f in filenames:
                # frankfurt  lindau  munster
                root_path = gt_path + '/' + f.split('_')[0]
                pattern = re.compile('\\w+_\\d+_\\d+')  # 跟新的规则使用两个\\表示，以前是一个
                gt_file = re.findall(pattern, f)[0] + '_gtFine_labelTrainIds.png'
                gt_file = os.path.join(root_path, gt_file)
                gt_files.append(gt_file)

        return gt_files

    def forward_eval(self, path=None):
        if path is None:
            path = self.work_dir
        filenames = []
        for file in mmengine.scandir(path, '.json', recursive=True):
            filenames.append(file)
        # filenames = [f for f in os.listdir(path) if '.json' in f]

        gt_files = self.get_gt_path(filenames, path)

        num_classes = len(self.CLASSES)
        pre_eval_results = []
        file_client = mmengine.FileClient(**{'backend': 'disk'})  # https://zhuanlan.zhihu.com/p/339190576

        for i, f in enumerate(tqdm(filenames)):
            result = mmengine.load(os.path.join(path, f))
            init_flag = True
            for id_str, mask in result['semantic_mask'].items():  # keys: size and counts
                mask_ = maskUtils.decode(mask)  # 按每个class处理的，感觉后面得优化一下这样效率不是很高
                h, w = mask_.shape
                if init_flag:
                    seg_mask = torch.zeros((1, 1, h, w))
                    init_flag = False
                mask_ = torch.from_numpy(mask_).unsqueeze(0).unsqueeze(0)
                seg_mask[mask_] = int(id_str)
            # 把所有类的结果整合到一个里面
            seg_logit = torch.zeros((1, num_classes, h, w))
            seg_logit.scatter_(1, seg_mask.long(), 1)
            seg_logit = seg_logit.float()
            # print(seg_logit.shape)
            # seg_pred = F.softmax(seg_logit, dim=1).argmax(dim=1).squeeze(0).numpy()
            seg_pred = seg_logit.argmax(dim=1).squeeze(0).numpy()

            # get gt
            gt_file = gt_files[i]
            img_bytes = file_client.get(gt_file)
            seg_map = mmcv.imfrombytes(
                img_bytes, flag='unchanged',
                backend='pillow').squeeze().astype(np.uint8)
            print(Counter(seg_pred.flatten()))
            print(Counter(seg_map.flatten()))
            pre_eval_results.append(intersect_and_union(
                seg_pred,
                seg_map,
                num_classes,
                255,
                label_map=dict(),
                reduce_zero_label=False))

        ret_metrics = pre_eval_to_metrics(pre_eval_results, ['mIoU'])
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        print(ret_metrics)

        print(ret_metrics_summary)