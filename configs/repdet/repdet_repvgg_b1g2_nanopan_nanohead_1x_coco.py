_base_ = [
    '../_base_/models/repdet_repvgg_pafpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_poly.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='RepDet',
    pretrained='/data/kartikes/repvgg_models/repvgg_b1g2.pth',
    backbone=dict(
        type='RepVGG',
        arch='B1g2',
        out_stages=[1, 2, 3, 4],
        activation='ReLU',
        last_channel=1024,
        deploy=False), 
    neck=dict(
        type='NanoPAN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=5,
        start_level=1,
        add_extra_convs='on_input'),
    bbox_head=dict(
        type='NanoDetHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
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
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0))
    )

optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0001)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2)
find_unused_parameters=True
runner = dict(type='EpochBasedRunner', max_epochs=12)
