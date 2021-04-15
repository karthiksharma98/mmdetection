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
        type='GFLHead',
        num_classes=80,
        in_channels=128,
        stacked_convs=2,
        feat_channels=128,
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
        reg_max=10,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
   
    # model training and testing settings
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
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
