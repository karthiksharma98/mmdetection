model = dict(
    type='RepDet',
    backbone=dict(
        type='RepVGG',
        arch=None,
        out_stages=[2, 3, 4],
        activation='ReLU',
        last_channel=512,
        deploy=False), 
    neck=dict(
        type='PAFPN',
        in_channels=[96, 192, 512],
        out_channels=128,
        num_outs=3),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=128,
        stacked_convs=4,
        feat_channels=128,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32]),
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
   
    # model training and testing settings
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
        max_per_img=100)
    )
