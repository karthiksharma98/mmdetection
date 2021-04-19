_base_ = [
    '../_base_/models/repdet_repvgg_pafpn.py',
    '../_base_/datasets/coco_detection_2.py',
    '../_base_/schedules/schedule_poly.py', '../_base_/default_runtime.py'
]

model = dict(
    pretrained='/data/kartikes/repvgg_models/repvgg_a0.pth',
    backbone=dict(
        arch='A0'
    ),
    neck=dict(
        type='NanoPAN',
        in_channels=[96, 192, 512],
        out_channels=128,
        num_outs=3),
    bbox_head=dict(
        type='NanoDetHead',
        num_classes=80,
        in_channels=128,
        stacked_convs=2,
        feat_channels=128,
        share_cls_reg=True,
        reg_max=10,
        norm_cfg=dict(type='BN', requires_grad=True),
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
)
optimizer = dict(type='SGD', lr=0.07, momentum=0.9, weight_decay=0.0001)
data = dict(
    samples_per_gpu=42,
    workers_per_gpu=1)
find_unused_parameters=True
