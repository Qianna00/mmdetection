import mmcv
from mmcv.runner import EpochBasedRunner
from mmcv.runner.hooks import IterTimerHook, HOOKS, Hook
from mmcv.runner.priority import get_priority
import torch
import os.path as osp
from mmcv.runner.checkpoint import save_checkpoint
import time
from mmcv.parallel import is_module_wrapper
from torch.optim import Optimizer
from mmcv.runner.checkpoint import weights_to_cpu


class MultiOptimRunner(EpochBasedRunner):

    def __init__(self,
                 model,
                 optimizer_b=None,
                 optimizer_g=None,
                 optimizer_d=None,
                 work_dir=None,
                 logger=None,
                 meta=None):
        self.optimizer_b = optimizer_b
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        # optimizer = {"optimizer_b": optimizer_b, "optimizer_g": optimizer_g, "optimizer_d": optimizer_d}
        super(MultiOptimRunner, self).__init__(model=model, work_dir=work_dir, logger=logger, meta=meta)

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
                param groups. If the runner has a dict of optimizers, this
                method will return a dict.
        """
        if isinstance(self.optimizer_b, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.optimizer_b.param_groups]
        elif isinstance(self.optimizer_b, dict):
            lr = dict()
            for name, optim in self.optimizer_b.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr

    def register_hook(self, hook, priority='HIGH'):
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :cls:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def register_lr_hook(self, lr_config, type='B'):
        if isinstance(lr_config, dict):
            assert 'policy' in lr_config
            policy_type = lr_config.pop('policy')
            # If the type of policy is all in lower case, e.g., 'cyclic',
            # then its first letter will be capitalized, e.g., to be 'Cyclic'.
            # This is for the convenient usage of Lr updater.
            # Since this is not applicable for `CosineAnealingLrUpdater`,
            # the string will not be changed if it contains capital letters.
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + 'LrUpdaterHook'
            lr_config['type'] = hook_type + type
            hook = mmcv.build_from_cfg(lr_config, HOOKS)
        else:
            hook = lr_config
        self.register_hook(hook)

    def register_momentum_hook(self, momentum_config):
        if momentum_config is None:
            return
        if isinstance(momentum_config, dict):
            assert 'policy' in momentum_config
            policy_type = momentum_config.pop('policy')
            # If the type of policy is all in lower case, e.g., 'cyclic',
            # then its first letter will be capitalized, e.g., to be 'Cyclic'.
            # This is for the convenient usage of momentum updater.
            # Since this is not applicable for `CosineAnealingMomentumUpdater`,
            # the string will not be changed if it contains capital letters.
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + 'MomentumUpdaterHook'
            momentum_config['type'] = hook_type
            hook = mmcv.build_from_cfg(momentum_config, HOOKS)
        else:
            hook = momentum_config
        self.register_hook(hook)

    def register_optimizer_hook(self, optimizer_config, priority='NORMAL', optim_type="OptimHookB"):
        if optimizer_config is None:
            return
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault('type', optim_type)
            hook = mmcv.build_from_cfg(optimizer_config, HOOKS)
        else:
            hook = optimizer_config
        self.register_hook(hook, priority)

    def register_training_hooks(self,
                                lr_config_b,
                                lr_config_g=None,
                                lr_config_d=None,
                                optimizer_b_config=None,
                                optimizer_g_config=None,
                                optimizer_d_config=None,
                                checkpoint_config=None,
                                log_config=None,
                                momentum_config=None):
        self.register_lr_hook(lr_config_b, type='B')
        self.register_lr_hook(lr_config_g, type='G')
        self.register_lr_hook(lr_config_d, type='D')
        self.register_momentum_hook(momentum_config)
        self.register_optimizer_hook(optimizer_b_config, priority="HIGH", optim_type="OptimHookB")
        self.register_optimizer_hook(optimizer_g_config, priority="LOW", optim_type="OptimHookG")
        self.register_optimizer_hook(optimizer_d_config, priority="NORMAL", optim_type="OptimHookD")
        # self.register_optimizer_hook(optimizer_b_config, priority="NORMAL", optim_type="MultiOptimHook")
        self.register_checkpoint_hook(checkpoint_config)
        self.register_hook(IterTimerHook())
        self.register_logger_hooks(log_config)

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer_b = self.optimizer_b if save_optimizer else None
        optimizer_g = self.optimizer_g if save_optimizer else None
        optimizer_d = self.optimizer_g if save_optimizer else None
        _save_checkpoint(self.model, filepath, optimizer_b=optimizer_b, optimizer_g=optimizer_g,
                         optimizer_d=optimizer_d, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            mmcv.symlink(filename, osp.join(out_dir, 'latest.pth'))

def _save_checkpoint(model, filename, optimizer_b=None, optimizer_g=None, optimizer_d=None, meta=None):
    """Save checkpoint to file.

    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default ``meta`` will contain version and time info.

    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f'meta must be a dict or None, but got {type(meta)}')
    meta.update(mmcv_version=mmcv.__version__, time=time.asctime())

    mmcv.mkdir_or_exist(osp.dirname(filename))
    if is_module_wrapper(model):
        model = model.module

    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(model.state_dict())
    }
    # save optimizer state dict in the checkpoint
    if isinstance(optimizer_b, Optimizer):
        checkpoint['optimizer_b'] = optimizer_b.state_dict()
    elif isinstance(optimizer_b, dict):
        checkpoint['optimizer_b'] = {}
        for name, optim in optimizer_b.items():
            checkpoint['optimizer_b'][name] = optim.state_dict()
    if isinstance(optimizer_g, Optimizer):
        checkpoint['optimizer_g'] = optimizer_g.state_dict()
    elif isinstance(optimizer_g, dict):
        checkpoint['optimizer_g'] = {}
        for name, optim in optimizer_g.items():
            checkpoint['optimizer_g'][name] = optim.state_dict()
    if isinstance(optimizer_d, Optimizer):
        checkpoint['optimizer_d'] = optimizer_d.state_dict()
    elif isinstance(optimizer_d, dict):
        checkpoint['optimizer_d'] = {}
        for name, optim in optimizer_d.items():
            checkpoint['optimizer_d'][name] = optim.state_dict()
    # immediately flush buffer
    with open(filename, 'wb') as f:
        torch.save(checkpoint, f)
        f.flush()
