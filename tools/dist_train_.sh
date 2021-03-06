#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 2001 \
    $(dirname "$0")/train_.py $CONFIG --launcher pytorch --work_dir /root/data/zq/mmdetection/outputs_temp \
    --resume-from /root/data/zq/mmdetection/outputs_temp/latest.pth ${@:3}
