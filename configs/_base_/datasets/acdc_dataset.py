# dataset settings
dataset_type = 'ACDCDataset'
data_root = './data/Segmentation/ACDC'
crop_size = (512, 1024)  # [W, H]  注意不是[H, W]
# scale = (1920//2, 1080//2)
scale = (512, 512)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=scale, keep_ratio=True),
    # 5112, 1024太大了
    # dict(type='Resize', scale=(1024, 512), keep_ratio=False),  # 1024也运行不了那个prompt嵌入
    # dict(type='Resize', scale=(512, 512), keep_ratio=False),  # 1024也运行不了那个prompt嵌入

    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform

    # dict(type='PhotoMetricDistortion'),
    dict(type='LoadAnnotations'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

# test time augmentation
# img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
img_ratios = [0.5, 1.0, 2.0]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    # tt augmentation这里就设为false就是之前的结果
    dict(type='Resize', scale=scale, keep_ratio=False),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=False)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]


test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='rgb_anon/fog/train', seg_map_path='gt/fog/train'),
        pipeline=test_pipeline))


test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])