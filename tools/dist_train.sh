#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 2041 \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --work_dir /root/vsislab-2/zq/smd_det/two_backbone_separate/imagenet_and_imagenet ${@:3}
