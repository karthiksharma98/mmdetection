#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

for img_size in {288,352,416,480,544,608,672,736,800,864}; do
    echo -e "\n"
    echo bash ./tools/dist_test.sh $CONFIG $CHECKPOINT $GPUS --eval bbox --fuse-conv-bn --cfg-options fp16="dict(loss_scale=512.)" --img_size $img_size ${@:4}
    bash ./tools/dist_test.sh $CONFIG $CHECKPOINT $GPUS --eval bbox --fuse-conv-bn --cfg-options fp16="dict(loss_scale=512.)" --img_size $img_size ${@:4}
done

echo -e "\n"
echo bash ./tools/dist_test.sh $CONFIG $CHECKPOINT $GPUS --eval bbox --fuse-conv-bn --cfg-options fp16="dict(loss_scale=512.)" ${@:4}
bash ./tools/dist_test.sh $CONFIG $CHECKPOINT $GPUS --eval bbox --fuse-conv-bn --cfg-options fp16="dict(loss_scale=512.)" ${@:4}