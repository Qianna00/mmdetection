#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 2001 \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --work_dir /root/data/zq/smd_det/cb/6c_new_split ${@:3}
