# optimizer
optimizer = dict(type='AdamW', lr=1e-3)
#optimizer = dict(type='RMSprop', lr=1e-3)
#optimizer = dict(type='SGD', lr=1e-1)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=1.0, min_lr=1e-6, by_epoch=False,
                warmup='linear', warmup_iters=4000, warmup_ratio=1e-6)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
#evaluation = dict(interval=4000, metric=['mIoU', 'mAuROC'])
evaluation = dict(interval=4000, metric=['mIoU', 'mFishy'])
