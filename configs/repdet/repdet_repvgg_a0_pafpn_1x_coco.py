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
