import math


class _LR:
    def __init__(self, optimizer, last_epoch):
        self._optimizer = optimizer
        self.last_epoch = last_epoch

    def _get_lr(self):
        raise NotImplementedError

    @property
    def current_lr(self):
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        lr = self._get_lr()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class ConstantCosineAnnealingLR(_LR):
    """LR scheduler that begins with a constant LR
    schedule before switching to a decaying LR schedule
    """
    def __init__(self, optimizer, total_epochs, init_lr, alpha=0.01, const_frac=0.5, last_epoch=-1):
        super().__init__(optimizer, last_epoch)
        self._optimizer = optimizer
        self.total_epochs = total_epochs
        self.switch_epoch = math.floor(total_epochs*const_frac)
        self.cosine_epochs = self.total_epochs - self.switch_epoch
        self.init_lr = init_lr
        self.final_lr = alpha*init_lr
        self._curr_lr = init_lr

    @property
    def current_lr(self):
        return self._curr_lr

    def _get_lr(self):
        if self.last_epoch > self.switch_epoch:
            epochs_since_switch = min(self.last_epoch - self.switch_epoch, self.cosine_epochs)
            decay = 1 + math.cos(epochs_since_switch*math.pi/self.cosine_epochs)
            lr = self.final_lr + 0.5*(self.init_lr - self.final_lr)*decay
        else:
            lr = self.init_lr
        self._curr_lr = lr
        return lr
