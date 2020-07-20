import mmcv
from mmcv.runner import EpochBasedRunner
from mmcv.runner.hooks import IterTimerHook, HOOKS, Hook
from mmcv.runner.priority import get_priority
import torch


class MultiOptimRunner(EpochBasedRunner):

    def __init__(self,
                 model,
                 batch_processor=None,
                 optimizer_b=None,
                 optimizer_g=None,
                 optimizer_d=None,
                 work_dir=None,
                 logger=None,
                 meta=None):
        optimizer = {"optimizer_b": optimizer_b, "optimizer_g": optimizer_g, "optimizer_d": optimizer_d}
        super(MultiOptimRunner, self).__init__(model=model, batch_processor=batch_processor, optimizer=optimizer,
                                               work_dir=work_dir, logger=logger, meta=meta)

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

    def register_optimizer_hook(self, optimizer_config, priority='NORMAL'):
        if optimizer_config is None:
            return
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault('type', 'OptimizerHook')
            hook = mmcv.build_from_cfg(optimizer_config, HOOKS)
        else:
            hook = optimizer_config
        self.register_hook(hook, priority)

    def register_training_hooks(self,
                                lr_config,
                                optimizer_b_config=None,
                                optimizer_g_config=None,
                                optimizer_d_config=None,
                                checkpoint_config=None,
                                log_config=None,
                                momentum_config=None):
        self.register_lr_hook(lr_config)
        self.register_momentum_hook(momentum_config)
        self.register_optimizer_hook(optimizer_b_config, priority="LOW")
        self.register_optimizer_hook(optimizer_g_config, priority="HIGH")
        self.register_optimizer_hook(optimizer_d_config, priority="NORMAL")
        self.register_checkpoint_hook(checkpoint_config)
        self.register_hook(IterTimerHook())
        self.register_logger_hooks(log_config)
