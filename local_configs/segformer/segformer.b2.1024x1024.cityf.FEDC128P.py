_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_flow.py',
    '../_base_/datasets/cityf_1024x1024.py'
]  # '../_base_/models/segformer.py',

# model settings
find_unused_parameters = True
norm_cfg = dict(type='SyncBN', momentum=0.0, track_running_stats=True)  # avoid updating BN running stats
embed_dim = 64
num_classes = 19
ignore_index = 255
num_classes = 19
ood_class_index_train = num_classes  # indices of ood classes for evaluation
ood_model = 'LOGITFLOWFMD'

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
        type='SegformerFlowHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.0,
        num_classes=num_classes,
        ignore_index=ignore_index,
        norm_cfg=norm_cfg,
        align_corners=False,
        flow_decode=dict(type='realnvp2d-cflow', model=ood_model, coupling_blocks=8, conditions=128),
        loss_decode=dict(type='FlowLoss', num_classes=num_classes, use_sigmoid=False),
        ood_class_index=ood_class_index_train,
        ood_est_stats=False,
        ood_dir_stats='stats/segformer_b2_1024x1024.cityf',
        freeze=True,
        flow_upsample=False,
        post_processing=True),
    # model training and testing settings
    train_cfg=dict(mode='flow_train'),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))
