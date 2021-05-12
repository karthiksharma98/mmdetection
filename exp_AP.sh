#!/usr/bin/env bash

# bash study_AP.sh /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_dilatedencoder_yolof_1x_coco/repdet_repvgg_a2_dilatedencoder_yolof_1x_coco.py /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_dilatedencoder_yolof_1x_coco/latest.pth 10 --convert-repvgg --eval-out map_rvgg_yolof.txt

# bash study_AP.sh /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_fpn_retinahead_1x_coco/repdet_repvgg_a2_fpn_retinahead_1x_coco.py /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_fpn_retinahead_1x_coco/latest.pth 10 --convert-repvgg --eval-out map_rvgg_fpn_retina.txt

# bash study_AP.sh /home/kartikes/repo/mmdet_ksharma/configs/yolof/yolof_r50_c5_8x8_1x_coco.py http://download.openmmlab.com/mmdetection/v2.0/yolof/yolof_r50_c5_8x8_1x_coco/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth 10 --eval-out map_yolof.txt

# bash study_AP.sh /home/kartikes/repo/mmdet_ksharma/configs/retinanet/retinanet_r50_fpn_1x_coco.py http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth 10 --eval-out map_retinanet.txt

# bash study_AP.sh /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_b1g2_dilatedencoder_yolof_1x_coco/repdet_repvgg_b1g2_dilatedencoder_yolof_1x_coco.py /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_b1g2_dilatedencoder_yolof_1x_coco/latest.pth 4 --convert-repvgg --eval-out map_rvgg_b1g2_yolof.txt

# bash study_AP.sh /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_b1g2_fpn_retinahead_1x_coco/repdet_repvgg_b1g2_fpn_retinahead_1x_coco.py /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_b1g2_fpn_retinahead_1x_coco/latest.pth 4 --convert-repvgg --eval-out map_rvgg_b1g2_fpn_retina.txt

bash study_AP.sh /home/kartikes/repo/mmdet_ksharma/configs/yolof/yolof_r101_c5_8x8_1x_coco.py  /home/kartikes/repo/mmdet_ksharma/work_dirs/yolof_r101_c5_8x8_1x_coco/latest.pth 10 --eval-out map_yolof_r101.txt

# bash study_AP.sh /home/kartikes/repo/mmdet_ksharma/configs/yolof/yolof_r101_c5_8x8_1x_coco.py  /home/kartikes/repo/mmdet_ksharma/work_dirs/yolof_r101_c5_8x8_1x_coco/latest.pth 4 --eval-out map_retinanet_r101.txt