_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_flow.py',
    '../_base_/datasets/cityf.py'
]  # '../_base_/models/deeplabv3plus_r101-d8.py',

# model settings
norm_cfg = dict(type='SyncBN', momentum=0.0, track_running_stats=True)  # avoid updating BN running stats
embed_dim = 512
ignore_index = 255
num_classes = 19
ood_class_index_train = num_classes  # indices of ood classes for evaluation
ood_model = 'LOGITFLOWFMD'

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
        type='DepthwiseSeparableFlowASPPHead',
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
        flow_decode=dict(type='realnvp2d-flow', model=ood_model, coupling_blocks=4, conditions=0),
        loss_decode=dict(type='FlowLoss', num_classes=num_classes, use_sigmoid=False),
        ood_class_index=ood_class_index_train,
        ood_est_stats=False,
        ood_dir_stats='stats/deeplabv3plus_r101-d8_512x1024.cityf',
        freeze=True,
        flow_upsample=False,
        post_processing=True),
    # model training and testing settings
    train_cfg=dict(mode='flow_train'),
    test_cfg=dict(mode='whole'))
