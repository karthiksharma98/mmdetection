#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2

for img_size in {288,352,416,480,544,608,672,736,800,864}; do
    echo -e "\n"
    echo CUDA_LAUNCH_BLOCKING=1 python ./tools/analysis_tools/benchmark.py $CONFIG $CHECKPOINT --img_size $img_size ${@:3}
    CUDA_LAUNCH_BLOCKING=1 python ./tools/analysis_tools/benchmark.py $CONFIG $CHECKPOINT --img_size $img_size ${@:3}
done

echo -e "\n"
echo CUDA_LAUNCH_BLOCKING=1 python ./tools/analysis_tools/benchmark.py $CONFIG $CHECKPOINT ${@:3}
CUDA_LAUNCH_BLOCKING=1 python ./tools/analysis_tools/benchmark.py $CONFIG $CHECKPOINT ${@:3}