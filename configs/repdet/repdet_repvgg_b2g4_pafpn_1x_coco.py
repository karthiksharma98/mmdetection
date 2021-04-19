_base_ = [
    '../_base_/models/repdet_repvgg_pafpn.py',
    '../_base_/datasets/coco_detection_2.py',
    '../_base_/schedules/schedule_poly.py', '../_base_/default_runtime.py'
]

model = dict(
    pretrained='/data/kartikes/repvgg_models/repvgg_b2g4.pth',
    backbone=dict(
        arch='B2g4',
        last_channel=1024,
        out_stages=[1, 2, 3, 4]),
    neck=dict(
        type='PAFPN',
        in_channels=[160, 320, 640, 1024],
        out_channels=512,
        num_outs=4),
    bbox_head=dict(
        type='GFLHead',
        num_classes=80,
        in_channels=512,
        stacked_convs=4,
        feat_channels=512,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
)
optimizer = dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=24)
load_from = '/home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_b2g4_pafpn_1x_coco/latest.pth'