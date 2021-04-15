_base_ = [
    '../_base_/models/repdet_repvgg_pafpn.py',
    '../_base_/datasets/coco_detection_2.py',
    '../_base_/schedules/schedule_poly.py', '../_base_/default_runtime.py'
]

model = dict(
    pretrained='/data/kartikes/repvgg_models/repvgg_a0.pth',
    backbone=dict(
        arch='A0'
    )
)

optimizer = dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=24)
resume_from = '/home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a0_pafpn_1x_coco/latest.pth'
