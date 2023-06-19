# dataset settings
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
ignore_index = 255
ood_class_index_train = ignore_index  # indices of ood classes for evaluation
img_suffix     = '_leftImg8bit.png'
seg_map_suffix = '_gtFine_labelTrainIds.png'
img_scale_test = (2048, 1024)
test_dataset = 'FSLF'
if test_dataset == 'FSST':  # Fishyscapes Static
    ood_class_index_test = [1] # indices of ood classes for evaluation
    data_test_root = 'data/fishyscapes/FsValV3Static'
elif test_dataset == 'FSLF':  # Fishyscapes Lost and Found
    ood_class_index_test = [1] # indices of ood classes for evaluation
    data_test_root = 'data/fishyscapes/FsValLF'
elif test_dataset == 'CS': # Cityscapes
    ood_class_index_test = [255] # indices of ood classes for evaluation
    data_test_root = data_root
elif test_dataset == 'SA':  # SegmentMeIfYouCan Anomaly
    ood_class_index_test = [1] # indices of ood classes for evaluation
    data_test_root = 'data/segmeifyoucan/AnomalyTrack'
    img_suffix     = '.jpg'
    seg_map_suffix = '_labels_semantic.png'
    img_scale_test = (1280, 720)
elif test_dataset == 'SO':  # SegmentMeIfYouCan Obstacle
    ood_class_index_test = [1] # indices of ood classes for evaluation
    data_test_root = 'data/segmeifyoucan/ObstacleTrack'
    img_suffix     = '.webp'
    seg_map_suffix = '_labels_semantic.png'
    img_scale_test = (1920, 1080)

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

city_crop_size = (512, 1024)  # for DLV3
city_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=city_crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=city_crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Corruptions', mode='test'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale_test,
        flip=False,
        #img_ratios=[0.5, 1.0, 1.5],  # used for CS TTA to get higher mIoU and open-mIoU
        img_ratios=[1.0, 0.5, 0.25],  # used for FS and SMIYC TTA experiments to improve OOD metrics + need to enable upsampling parameter flow_upsample=True in flow_decode_head.py
        #flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

city_train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    img_dir='leftImg8bit/train',
    ann_dir='gtFine/train',
    pipeline=city_train_pipeline,
    img_suffix='_leftImg8bit.png',
    seg_map_suffix='_gtFine_labelTrainIds.png')

val_dataset = dict(
    type=dataset_type,
    data_root=data_test_root,
    img_dir='leftImg8bit/val',
    ann_dir='gtFine/val',
    pipeline=test_pipeline,
    img_suffix=img_suffix,
    seg_map_suffix=seg_map_suffix,
    ood_class_index=ood_class_index_test,
    ignore_index=ignore_index)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=city_train_dataset,
    val=val_dataset,
    test=val_dataset)
