#!/usr/bin/env bash

OUT=ablation_rvgg_fpn.txt

# echo "\nrvgg_fpn_retinahead vanilla" >> $OUT
# python ./tools/analysis_tools/timing_fp16.py --config /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_fpn_retinahead_1x_coco/repdet_repvgg_a2_fpn_retinahead_1x_coco.py --checkpoint /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_fpn_retinahead_1x_coco/latest.pth --eval-out $OUT

# echo "\nrvgg_fpn_retinahead reparam" >> $OUT
# python ./tools/analysis_tools/timing_fp16.py --config /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_fpn_retinahead_1x_coco/repdet_repvgg_a2_fpn_retinahead_1x_coco.py --checkpoint /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_fpn_retinahead_1x_coco/latest.pth --convert-repvgg --eval-out $OUT

# echo "\nrvgg_fpn_retinahead fp16 reparam" >> $OUT
# python ./tools/analysis_tools/timing_fp16.py --config /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_fpn_retinahead_1x_coco/repdet_repvgg_a2_fpn_retinahead_1x_coco.py --checkpoint /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_fpn_retinahead_1x_coco/latest.pth --convert-repvgg --cfg-options fp16="dict(loss_scale=512.)" --eval-out $OUT

# echo "\nrvgg_fpn_retinahead fuse_conv_bn fp16 reparam" >> $OUT
# python ./tools/analysis_tools/timing_fp16.py --config /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_fpn_retinahead_1x_coco/repdet_repvgg_a2_fpn_retinahead_1x_coco.py --checkpoint /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_fpn_retinahead_1x_coco/latest.pth --convert-repvgg --fuse-conv-bn --cfg-options fp16="dict(loss_scale=512.)" --eval-out $OUT

echo "\nrvgg_fpn_retinahead fp16 vanilla" >> $OUT
python ./tools/analysis_tools/timing_fp16.py --config /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_fpn_retinahead_1x_coco/repdet_repvgg_a2_fpn_retinahead_1x_coco.py --checkpoint /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_fpn_retinahead_1x_coco/latest.pth --cfg-options fp16="dict(loss_scale=512.)" --eval-out $OUT

###

OUT=ablation_rvgg_yolof.txt

# echo "\nrvgg_yolof vanilla" >> $OUT
# python ./tools/analysis_tools/timing_fp16.py --config /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_dilatedencoder_yolof_1x_coco/repdet_repvgg_a2_dilatedencoder_yolof_1x_coco.py --checkpoint /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_dilatedencoder_yolof_1x_coco/latest.pth --eval-out $OUT

# echo "\nrvgg_yolof reparam" >> $OUT
# python ./tools/analysis_tools/timing_fp16.py --config /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_dilatedencoder_yolof_1x_coco/repdet_repvgg_a2_dilatedencoder_yolof_1x_coco.py --checkpoint /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_dilatedencoder_yolof_1x_coco/latest.pth --convert-repvgg --eval-out $OUT

# echo "\nrvgg_yolof fp16 reparam" >> $OUT
# python ./tools/analysis_tools/timing_fp16.py --config /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_dilatedencoder_yolof_1x_coco/repdet_repvgg_a2_dilatedencoder_yolof_1x_coco.py --checkpoint /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_dilatedencoder_yolof_1x_coco/latest.pth --convert-repvgg --cfg-options fp16="dict(loss_scale=512.)" --eval-out $OUT

# echo "\nrvgg_yolof fuse_conv_bn fp16 reparam" >> $OUT
# python ./tools/analysis_tools/timing_fp16.py --config /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_dilatedencoder_yolof_1x_coco/repdet_repvgg_a2_dilatedencoder_yolof_1x_coco.py --checkpoint /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_dilatedencoder_yolof_1x_coco/latest.pth --convert-repvgg --fuse-conv-bn --cfg-options fp16="dict(loss_scale=512.)" --eval-out $OUT

echo "\nrvgg_yolof fp16 vanilla" >> $OUT
python ./tools/analysis_tools/timing_fp16.py --config /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_dilatedencoder_yolof_1x_coco/repdet_repvgg_a2_dilatedencoder_yolof_1x_coco.py --checkpoint /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_dilatedencoder_yolof_1x_coco/latest.pth --cfg-options fp16="dict(loss_scale=512.)" --eval-out $OUT