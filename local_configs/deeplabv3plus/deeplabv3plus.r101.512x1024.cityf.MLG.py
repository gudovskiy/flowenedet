_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_norm.py',
    '../_base_/datasets/cityf.py'
]  # '../_base_/models/deeplabv3plus_r101-d8.py',

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
embed_dim = 512
num_classes = 19
ood_class_index_train = [] # indices of ood classes for training
class_weight = [0.0 if cl in ood_class_index_train else 1.0 for cl in range(num_classes)]  # for training
ignore_index = 255

model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=embed_dim,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.0,
        num_classes=num_classes,
        ignore_index=ignore_index,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, class_weight=class_weight, loss_weight=1.0),
        ood_est_stats=False,
        ood_dir_stats='stats/deeplabv3plus_r101-d8_512x1024.cityf',
        ood_type='MLG',
        freeze=True,
        post_processing=False),
    # model training and testing settings
    train_cfg=dict(mode='norm_train'),
    test_cfg=dict(mode='whole'))
