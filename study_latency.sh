#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2

for img_size in {288,352,416,480,544,608,672,736,800,864}; do
    echo -e "\n"
    echo python ./tools/analysis_tools/timing_fp16.py --config $CONFIG --checkpoint $CHECKPOINT --fuse-conv-bn --cfg-options fp16="dict(loss_scale=512.)" --img_size $img_size ${@:3}
    python ./tools/analysis_tools/timing_fp16.py --config $CONFIG --checkpoint $CHECKPOINT --fuse-conv-bn --cfg-options fp16="dict(loss_scale=512.)" --img_size $img_size ${@:3}
done

echo -e "\n"
echo python ./tools/analysis_tools/timing_fp16.py --config $CONFIG --checkpoint $CHECKPOINT --fuse-conv-bn --cfg-options fp16="dict(loss_scale=512.)" ${@:3}
python ./tools/analysis_tools/timing_fp16.py --config $CONFIG --checkpoint $CHECKPOINT --fuse-conv-bn --cfg-options fp16="dict(loss_scale=512.)" ${@:3}