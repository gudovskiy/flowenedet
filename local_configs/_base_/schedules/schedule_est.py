# optimizer
optimizer = dict(type='AdamW', lr=0.0)
#optimizer = dict(type='RMSprop', lr=1e-3)
#optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, nesterov=True)
optimizer_config = dict()
# learning policy & runtime settings:
lr_config = dict(policy='step', step=15000, min_lr=1e-6, by_epoch=False, warmup='linear', warmup_iters=4000, warmup_ratio=1e-6)
runner = dict(type='IterBasedRunner', max_iters=1000)
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(interval=5000, metric=['mIoU', 'mOurs'])
