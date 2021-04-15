# optimizer
optimizer = dict(type='SGD', lr=0.07, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='poly', # poly with power=1 reduces to linear decay
    by_epoch=False,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001)
runner = dict(type='EpochBasedRunner', max_epochs=12)
