import torch
from tqdm import tqdm
from torch import nn, optim
import torch.nn.functional as F
from copy import deepcopy
from .sourceonly import SourceOnly, CHECK_NUM_PARAMS, create_ema_model


def get_source_dataloader(cfg, runner):
    train_dataloader_cfg = deepcopy(cfg.test_dataloader)
    train_dataloader_cfg.dataset.type = 'CityscapesDataset'
    train_dataloader_cfg.dataset.data_root = 'data/Segmentation/Cityscapes'
    train_dataloader_cfg.dataset.data_prefix.img_path = 'leftImg8bit/train'
    train_dataloader_cfg.dataset.data_prefix.seg_map_path = 'gtFine/train'
    train_dataloader_cfg.dataset.pipeline[1]['keep_ratio'] = False  # 不然ACDC与CS的shape对不上
    train_dataloader_cfg.dataset.pipeline[1].scale = (1024, 512)
    train_dataloader_cfg.batch_size = 1
    train_dataloader = runner.build_dataloader(dataloader=train_dataloader_cfg)
    return train_dataloader


class TTAP(SourceOnly):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.source_training()
        # if self.cfg.domain == 'gta2cs':
        #     self.model.load_state_dict(torch.load('./pretrain_1e-05_gta2cs.pth'))
        # elif self.cfg.domain == 'synthia2cs':
        #     self.model.load_state_dict(torch.load('./pretrain_1e-05_syn2cs.pth'))
        self.model_ema = create_ema_model(self.model)

    def source_training(self, lr=1e-5):
        self.model.eval()
        optimizer = optim.Adam(self.model.parameters(), lr, weight_decay=1e-4)
        train_dataloader = get_source_dataloader(self.cfg, self.runner)
        for i, data in enumerate(tqdm(train_dataloader)):
            y = data['data_samples'][0].gt_sem_seg.data
            if i < 2095:
                outputs = self.model.test_step(data)
                logits = outputs[0].seg_logits.data.unsqueeze(0)
                loss = F.cross_entropy(logits, y.cuda(), ignore_index=255)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                break

        torch.save(self.model.state_dict(), './pretrain_{}_syn2cs.pth'.format(lr))

    def build_optimizer(self, lr=1e-6):
        params = []
        for n, p in self.model.named_parameters():
            if 'attn' in n and 'block2' in n:
                p.requires_grad = True
                params.append(p)
            elif 'prompt' in n:
                p.requires_grad = True
                params.append(p)
            else:
                p.requires_grad = False

        optimizer = optim.AdamW(self.model.parameters(), lr, weight_decay=1e-5)
        CHECK_NUM_PARAMS(self.model, lr)
        return optimizer

    def ema_update(self, alpha=0.99):
        for ema_param, param in zip(self.model_ema.parameters(), self.model.parameters()):
            ema_param.data[:] = alpha * ema_param[:].data[:] + (1 - alpha) * param[:].data[:]

    def forward_epoch(self, epoch, data, optimizer=None):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model_ema.test_step(data)  # bs =1
            pl = outputs[0].pred_sem_seg.data

        seg_logits = self.model.test_step(data)[0].seg_logits.data.unsqueeze(0)
        loss = F.cross_entropy(seg_logits, pl, ignore_index=255)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            outputs = self.model_ema.test_step(data)
        self.ema_update()
        return outputs

