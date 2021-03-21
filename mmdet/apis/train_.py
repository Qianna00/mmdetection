import random
from collections import OrderedDict


import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, OptimizerHook, Runner,
                         build_optimizer)
from ..mmcv.multi_optim_hook import OptimHookB, OptimHookG, OptimHookD
from mmdet.core import DistEvalHook, EvalHook, Fp16OptimizerHook
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.utils import get_root_logger
from ..mmcv.multi_optim_runner import MultiOptimRunner
from mmcv.runner.checkpoint import save_checkpoint


def train_detector_m(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    params_b = []
    """for key, value in dict(model.backbone.named_parameters()).items():
        if value.requires_grad:
            params_b += [{'params': [value]}]
    for key, value in dict(model.rpn_head.named_parameters()).items():
        if value.requires_grad:
            params_b += [{'params': [value]}]"""
    for key, value in dict(model.roi_head.shared_head.named_parameters()).items():
        if value.requires_grad:
            params_b += [{'params': [value]}]
    for key, value in dict(model.roi_head.bbox_head.named_parameters()).items():
        if value.requires_grad:
            params_b += [{'params': [value]}]
    optimizer_b = torch.optim.SGD(params_b, lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum,
                                  weight_decay=cfg.optimizer.weight_decay)
    optimizer_g = None
    optimizer_d = None
    if model.roi_head.with_fsr_generator:
        params_g = []
        for key, value in dict(model.roi_head.fsr_generator.named_parameters()).items():
            if value.requires_grad:
                params_g += [{'params': [value]}]
        for key, value in dict(model.roi_head.shared_head.named_parameters()).items():
            if value.requires_grad:
                params_g += [{'params': [value], 'lr': cfg.optimizer_b.lr}]
        for key, value in dict(model.roi_head.bbox_head.named_parameters()).items():
            if value.requires_grad:
                params_g += [{'params': [value], 'lr': cfg.optimizer_b.lr}]
        optimizer_g = torch.optim.SGD(params_g, lr=cfg.optimizer_g.lr, momentum=cfg.optimizer_g.momentum,
                                      weight_decay=cfg.optimizer_g.weight_decay)
    if model.roi_head.with_dis_head:
        params_d = []
        for key, value in dict(model.roi_head.dis_head.named_parameters()).items():
            if value.requires_grad:
                params_d += [{'params': [value]}]
        optimizer_d = torch.optim.SGD(params_d, lr=cfg.optimizer_d.lr, momentum=cfg.optimizer_d.momentum,
                                      weight_decay=cfg.optimizer_d.weight_decay)

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    runner = MultiOptimRunner(
        model,
        optimizer_b,
        optimizer_g,
        optimizer_d,
        cfg.work_dir,
        logger=logger,
        meta=meta)

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config_b = Fp16OptimizerHook(
            **cfg.optimizer_config_b, **fp16_cfg, distributed=distributed)
        optimizer_config_g = Fp16OptimizerHook(
            **cfg.optimizer_config_g, **fp16_cfg, distributed=distributed)
        optimizer_config_d = Fp16OptimizerHook(
            **cfg.optimizer_config_d, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config_b:
        optimizer_config_b = OptimHookB(**cfg.optimizer_config_b)
        optimizer_config_g = OptimHookG(**cfg.optimizer_config_g)
        optimizer_config_d = OptimHookD(**cfg.optimizer_config_d)
    else:
        optimizer_config_b = cfg.optimizer_config_b
        optimizer_config_g = cfg.optimizer_config_g
        optimizer_config_d = cfg.optimizer_config_d

    # register hooks
    runner.register_training_hooks(cfg.lr_config_b, cfg.lr_config_g, cfg.lr_config_d, optimizer_config_b,
                                   optimizer_config_g, optimizer_config_d,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
