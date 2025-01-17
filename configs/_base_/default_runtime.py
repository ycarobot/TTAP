default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# 这里我把mmsegmentation直接uninstall了，所以加载vis_backends的时候不知道为啥会有问题
# vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'ERROR'
load_from = None
resume = False

tta_model = dict(type='SegTTAModel')
