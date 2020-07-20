import random
from collections import OrderedDict


import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, OptimizerHook, Runner,
                         build_optimizer)

from mmdet.core import DistEvalHook, EvalHook, Fp16OptimizerHook
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.utils import get_root_logger
from ..mmcv.multi_optim_runner import MultiOptimRunner
from mmcv.runner.checkpoint import save_checkpoint


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars


def parse_losses_m(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        print(loss_name)
        if isinstance(loss_value, torch.Tensor):
            print(loss_value.shape)
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            print(len(loss_value), loss_value[0].shape)
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars


def batch_processor(model, data, train_mode):
    """Process a data batch.

    This method is required as an argument of Runner, which defines how to
    process a data batch and obtain proper outputs. The first 3 arguments of
    batch_processor are fixed.

    Args:
        model (nn.Module): A PyTorch model.
        data (dict): The data batch in a dict.
        train_mode (bool): Training mode or not. It may be useless for some
            models.

    Returns:
        dict: A dict containing losses and log vars.
    """
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def train_detector(model,
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
    for key, value in dict(model.backbone.named_parameters()).items():
        if value.requires_grad:
            params_b += [{'params': [value]}]
    for key, value in dict(model.rpn_head.named_parameters()).items():
        if value.requires_grad:
            params_b += [{'params': [value]}]
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
        optimizer_g = torch.optim.SGD(params_g, lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum,
                                      weight_decay=cfg.optimizer.weight_decay)
    if model.roi_head.with_dis_head:
        params_d = []
        for key, value in dict(model.roi_head.dis_head.named_parameters()).items():
            if value.requires_grad:
                params_d += [{'params': [value]}]
        optimizer_d = torch.optim.SGD(params_d, lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum,
                                      weight_decay=cfg.optimizer.weight_decay)

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
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
        batch_processor,
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
        optimizer_config_b = OptimizerHook(**cfg.optimizer_config_b)
        optimizer_config_g = OptimizerHook(**cfg.optimizer_config_g)
        optimizer_config_d = OptimizerHook(**cfg.optimizer_config_d)
    else:
        optimizer_config_b = cfg.optimizer_config_b
        optimizer_config_g = cfg.optimizer_config_g
        optimizer_config_d = cfg.optimizer_config_d

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config_b, optimizer_config_g, optimizer_config_d,
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
