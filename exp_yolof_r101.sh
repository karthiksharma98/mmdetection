#!/usr/bin/env bash

bash tools/dist_train.sh /home/kartikes/repo/mmdet_ksharma/configs/yolof/yolof_r101_c5_8x8_1x_coco.py 4
bash exp_AP.sh
bash exp_latency.sh