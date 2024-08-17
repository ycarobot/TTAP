import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import sys  # 记得每次都要把路径加上，不要import同名的module有问题
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
current_path = os.path.dirname(__file__)
import torch
import numpy as np
import random
from mmengine.config import Config, DictAction
from mmengine.runner import Runner, load_checkpoint
from TTA.main_train import refine_dataset, build_solver, refine_baseline
from mmengine.dataset import worker_init_fn


def parse_args():
    parser = argparse.ArgumentParser(description='Test-time adaptation')
    parser.add_argument('--config', help='train config file path', default='configs/tta/train_acdc.py')
    parser.add_argument('--domain', help='domain', default='fog')
    parser.add_argument('--baseline', help='train baseline', default='SourceOnly')
    parser.add_argument('--tta', help='test time augmentation', action='store_true')
    parser.add_argument('--seed', help='random_seed', default=0)
    parser.add_argument('--lr', help='learning rate', default=3e-5, type=float)
    parser.add_argument('--checkpoint', help='checkpoint', default=None, type=str)
    args = parser.parse_args()
    return args


def main(worker_seed=0):
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.baseline = args.baseline
    cfg.domain = args.domain
    cfg = refine_dataset(cfg)
    cfg = refine_baseline(cfg)
    if args.tta:  # test-time augmentation
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    if args.checkpoint is not None:
        cfg.model.backbone.init_cfg.checkpoint = args.checkpoint
        cfg.checkpoint = args.checkpoint
    cfg.work_dir = os.path.join('./Run_tta', cfg.baseline, cfg.domain)
    print('save to:', cfg.work_dir)
    os.makedirs(cfg.work_dir, exist_ok=True)
    runner = Runner.from_cfg(cfg)
    load_checkpoint(runner.model, cfg.checkpoint,
                    revise_keys=[(r'^module\.', ''), ('^ema_model\.', ''), ('model.', '')])
    solver = build_solver(cfg.baseline)(
        cfg=cfg,
        model=runner.model,
        dataloader=runner.test_dataloader,
        work_dir=cfg.work_dir,
        evaluator=runner.test_evaluator,
        runner=runner,
        efficient_test=False,   # 升级了用这个很快
        lrs=[args.lr]  # token-wise prompt weight
    )
    solver.forward_train()
    # solver.forward_eval()

    # torch.save(runner.model.state_dict(), './model_daformer_pretrain_{}_EVP_low_high.pth'.format(cfg.domain))


if __name__ == '__main__':
    worker_seed = 0
    worker_init_fn(0, 0, 0, seed=worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    main()