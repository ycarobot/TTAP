from mmseg.apis.baselines import (
    SourceOnly, TTAP)


def build_solver(baseline):
    solver_dicts = {
        'SourceOnly': SourceOnly,
        'TTAP': TTAP,
    }
    return solver_dicts[baseline]


def refine_baseline(cfg):
    baseline = cfg.baseline
    if baseline == 'TTAP':
        if cfg.domain in ['gta2cs', 'synthia2cs']:
            cfg.model.backbone.type = 'DAFormer_TTAP'

    return cfg


def refine_dataset(cfg):
    if cfg.test_dataloader.dataset.type == 'ACDCDataset':
        domain = cfg.test_dataloader.dataset.data_prefix.img_path.split('/')[-2]
        cfg.test_dataloader.dataset.data_prefix.img_path = (
            cfg.test_dataloader.dataset.data_prefix.img_path.replace(domain, cfg.domain))
        cfg.test_dataloader.dataset.data_prefix.seg_map_path = (
            cfg.test_dataloader.dataset.data_prefix.seg_map_path.replace(domain, cfg.domain))

    if cfg.test_dataloader.dataset.type == 'NTHUDataset':
        cfg.test_dataloader.dataset.data_prefix.img_path = '{}/Images/Test'.format(cfg.domain)
        cfg.test_dataloader.dataset.data_prefix.seg_map_path = '{}/Labels/Test'.format(cfg.domain)

    return cfg
