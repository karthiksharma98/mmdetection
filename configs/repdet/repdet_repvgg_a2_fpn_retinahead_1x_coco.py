_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_poly.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='RepDet',
    pretrained='/data/kartikes/repvgg_models/repvgg_a2.pth',
    backbone=dict(
        type='RepVGG',
        arch='A2',
        out_stages=[1, 2, 3, 4],
        activation='ReLU',
        last_channel=1024,
        deploy=False), 
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 1024],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0001)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2)
find_unused_parameters=True
runner = dict(type='EpochBasedRunner', max_epochs=12)
