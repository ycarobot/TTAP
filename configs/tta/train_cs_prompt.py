_base_ = [
    '../_base_/models/daformer.py',
    '../_base_/datasets/cs_dataset.py',
    '../_base_/default_runtime.py'
]

checkpoint = './data/Pth/Segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'  # noqa

# test_dataloader = dict(batch_size=1, num_workers=4)

test_cfg = dict(type='TestLoop')
# model = dict(
#     backbone=dict(
#         init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
#         embed_dims=64,
#         num_layers=[3, 6, 40, 3]),
#     decode_head=dict(in_channels=[64, 128, 320, 512]))

# set visual prompt
model = dict(
# type='MultiscaleEncoderDecoderPrompt',
    backbone=dict(
        # type='mit_b5_daformer_EVP',  # mit_b5_daformer_EVP_low_high, MiT_VPT, MiT_VPT_dense, MiT_VPT_sparse, MiT_EVP
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        num_layers=[3, 6, 40, 3],
        deep_prompt=False,),
        # decode head这里要改一下记得，不然也是对不上的
        decode_head=dict(
            decoder_params=dict(
                fusion_cfg=dict(
                    _delete_=True,
                    type='aspp',
                    sep=True,
                    dilations=(1, 6, 12, 18),
                    pool=False,
                    act_cfg=dict(type='ReLU'),
                    norm_cfg=dict(type='BN', requires_grad=True))))
)

test_dataloader = dict(
    batch_size=1)