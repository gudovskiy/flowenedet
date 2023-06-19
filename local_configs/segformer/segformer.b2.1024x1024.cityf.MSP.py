_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k_adamw.py',
    '../_base_/datasets/cityf_1024x1024.py'
]  # '../_base_/models/segformer.py',

# model settings
find_unused_parameters = True
norm_cfg = dict(type='SyncBN', requires_grad=True)
embed_dim = 64
num_classes = 19
ignore_index = 255
ood_class_index_train = [255]  # indices of ood classes for evaluation

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa

# model settings
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=embed_dim,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.0,
        num_classes=num_classes,
        ignore_index=ignore_index,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ood_est_stats=False,
        ood_dir_stats='stats/segformer_b2_1024x1024.cityf',
        ood_type='MSP',
        freeze=True,
        post_processing=False),
    # model training and testing settings
    train_cfg=dict(mode='flow_train'),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))
