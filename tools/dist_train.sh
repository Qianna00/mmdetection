#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 2032 \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --work_dir /root/vsislab-2/zq/smd_det/normal/faster_rcnn_5c_class_weighted ${@:3}
