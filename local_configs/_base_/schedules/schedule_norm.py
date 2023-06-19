# optimizer
optimizer = dict(type='AdamW', lr=1e-3)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=1.0, min_lr=1e-6, by_epoch=False,
                warmup='linear', warmup_iters=1000, warmup_ratio=1e-6)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=12000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric=['mIoU', 'mAuROC'])
