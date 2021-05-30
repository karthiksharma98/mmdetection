# echo "Timing 416"
# CUDA_LAUNCH_BLOCKING=1 python tools/analysis_tools/benchmark.py /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_fpn_retinahead_1x_coco_416x416/repdet_repvgg_a2_fpn_retinahead_1x_coco_416x416.py /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_fpn_retinahead_1x_coco_416x416/latest.pth --convert-repvgg

# echo "Timing 512"
# CUDA_LAUNCH_BLOCKING=1 python tools/analysis_tools/benchmark.py /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_fpn_retinahead_1x_coco_512x512/repdet_repvgg_a2_fpn_retinahead_1x_coco_512x512.py /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_fpn_retinahead_1x_coco_512x512/latest.pth --convert-repvgg

# echo "Timing 768"
# CUDA_LAUNCH_BLOCKING=1 python tools/analysis_tools/benchmark.py /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_fpn_retinahead_1x_coco_768x768/repdet_repvgg_a2_fpn_retinahead_1x_coco_768x768.py /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_fpn_retinahead_1x_coco_768x768/latest.pth --convert-repvgg

# echo "Timing 896"
# CUDA_LAUNCH_BLOCKING=1 python tools/analysis_tools/benchmark.py /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_fpn_retinahead_1x_coco_896x896/repdet_repvgg_a2_fpn_retinahead_1x_coco_896x896.py /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_fpn_retinahead_1x_coco_896x896/latest.pth --convert-repvgg

# echo "Timing default"
# CUDA_LAUNCH_BLOCKING=1 python tools/analysis_tools/benchmark.py /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_fpn_retinahead_1x_coco/repdet_repvgg_a2_fpn_retinahead_1x_coco.py /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_a2_fpn_retinahead_1x_coco/latest.pth --convert-repvgg

# echo "Timing retinanet"
# CUDA_LAUNCH_BLOCKING=1 python tools/analysis_tools/benchmark.py /home/kartikes/repo/mmdet_ksharma/configs/retinanet/retinanet_r50_fpn_1x_coco.py http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth

# echo "Timing b1g2"
# CUDA_LAUNCH_BLOCKING=1 python tools/analysis_tools/benchmark.py /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_b1g2_fpn_retinahead_1x_coco/repdet_repvgg_b1g2_fpn_retinahead_1x_coco.py /home/kartikes/repo/mmdet_ksharma/work_dirs/repdet_repvgg_b1g2_fpn_retinahead_1x_coco/latest.pth --convert-repvgg

echo "Timing a2 without convert"
CUDA_LAUNCH_BLOCKING=1 python tools/analysis_tools/benchmark.py configs/repdet/repdet_repvgg_a2_repdilatedencoder_yolof_1x_coco.py work_dirs/repdet_repvgg_a2_repdilatedencoder_yolof_1x_coco/latest.pth


echo "Timing a2 convert rep backbone"
CUDA_LAUNCH_BLOCKING=1 python tools/analysis_tools/benchmark.py configs/repdet/repdet_repvgg_a2_repdilatedencoder_yolof_1x_coco.py work_dirs/repdet_repvgg_a2_repdilatedencoder_yolof_1x_coco/latest.pth --convert-repvgg

echo "Timing a2 convert rep backbone + rep neck"
CUDA_LAUNCH_BLOCKING=1 python tools/analysis_tools/benchmark.py configs/repdet/repdet_repvgg_a2_repdilatedencoder_yolof_1x_coco.py work_dirs/repdet_repvgg_a2_repdilatedencoder_yolof_1x_coco/latest.pth --convert-repvgg --deploy_neck

# echo "Timing b1g2"
# CUDA_LAUNCH_BLOCKING=1 python tools/analysis_tools/benchmark.py work_dirs/repdet_repvgg_b1g2_dilatedencoder_yolof_1x_coco/repdet_repvgg_b1g2_dilatedencoder_yolof_1x_coco.py work_dirs/repdet_repvgg_b1g2_dilatedencoder_yolof_1x_coco/latest.pth

echo "Timing b1g2 without convert"
CUDA_LAUNCH_BLOCKING=1 python tools/analysis_tools/benchmark.py work_dirs/repdet_repvgg_b1g2_repdilatedencoder_yolof_1x_coco/repdet_repvgg_b1g2_repdilatedencoder_yolof_1x_coco.py work_dirs/repdet_repvgg_b1g2_repdilatedencoder_yolof_1x_coco/latest.pth

echo "Timing b1g2 convert rep backbone"
CUDA_LAUNCH_BLOCKING=1 python tools/analysis_tools/benchmark.py work_dirs/repdet_repvgg_b1g2_repdilatedencoder_yolof_1x_coco/repdet_repvgg_b1g2_repdilatedencoder_yolof_1x_coco.py work_dirs/repdet_repvgg_b1g2_repdilatedencoder_yolof_1x_coco/latest.pth --convert-repvgg

echo "Timing b1g2 convert rep backbone + rep neck"
CUDA_LAUNCH_BLOCKING=1 python tools/analysis_tools/benchmark.py work_dirs/repdet_repvgg_b1g2_repdilatedencoder_yolof_1x_coco/repdet_repvgg_b1g2_repdilatedencoder_yolof_1x_coco.py work_dirs/repdet_repvgg_b1g2_repdilatedencoder_yolof_1x_coco/latest.pth --convert-repvgg --deploy_neck

echo "Timing b1g2 convert rep backbone + rep neck + rep head"
CUDA_LAUNCH_BLOCKING=1 python tools/analysis_tools/benchmark.py work_dirs/repdet_repvgg_b1g2_repdilatedencoder_yolof_1x_coco/repdet_repvgg_b1g2_repdilatedencoder_yolof_1x_coco.py work_dirs/repdet_repvgg_b1g2_repdilatedencoder_yolof_1x_coco/latest.pth --convert-repvgg --deploy_neck --deploy_head

# nohup python tools/analysis_tools/get_flops.py  --model repblock --dilation 1 --shape 2048 56 56 >> test_block_flop_time.out

# nohup python tools/analysis_tools/get_flops.py  --model repblock --dilation 2  --shape 2048 56 56 >> test_block_flop_time.out

# nohup python tools/analysis_tools/get_flops.py  --model repblock --dilation 4 >> test_block_flop_time.out

# nohup python tools/analysis_tools/get_flops.py  --model repblock --dilation 8 >> test_block_flop_time.out

# nohup python tools/analysis_tools/get_flops.py  --model repblock --dilation 1 --deploy True >> test_block_flop_time.out

# nohup python tools/analysis_tools/get_flops.py  --model repblock --dilation 2 --deploy True >> test_block_flop_time.out

# nohup python tools/analysis_tools/get_flops.py  --model repblock --dilation 4 --deploy True >> test_block_flop_time.out

# nohup python tools/analysis_tools/get_flops.py  --model repblock --dilation 8 --deploy True >> test_block_flop_time.out

# nohup python tools/analysis_tools/get_flops.py  --model bottleneck --dilation 1 >> test_block_flop_time.out

# nohup python tools/analysis_tools/get_flops.py  --model bottleneck --dilation 2 >> test_block_flop_time.out

# nohup python tools/analysis_tools/get_flops.py  --model bottleneck --dilation 4 >> test_block_flop_time.out

# nohup python tools/analysis_tools/get_flops.py  --model bottleneck --dilation 8 >> test_block_flop_time.out


