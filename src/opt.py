# LR scheduler, unsupervised learning seems ill conditioned, needs greedy lr increases...
from torch.optim.lr_scheduler import _LRScheduler


class AdaptiveLR(object):
    """
    A learning rate scheduler that increases the learning rate when the loss
    is decreasing and decreases it when the loss is stagnating or increasing.
    """
    def __init__(self, optimizer, mode='min', factor=0.5, increase_factor=1.1, 
                 patience_increase=1, patience_decrease=2, min_lr=1e-6, 
                 max_lr=1e0, verbose=False):
        
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.increase_factor = increase_factor
        self.patience_increase = patience_increase
        self.patience_decrease = patience_decrease
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.verbose = verbose

        self.best_loss = float('inf') if self.mode == 'min' else float('-inf')
        self.num_bad_epochs = 0
        self.num_good_epochs = 0
        
        self.last_epoch = 0

    def step(self, metrics):
        current_loss = metrics

        if self.mode == 'min':
            if current_loss < self.best_loss:
                # Loss is improving, reset bad epochs counter and increment good epochs
                self.best_loss = current_loss
                self.num_bad_epochs = 0
                self.num_good_epochs += 1
                
                # Check if we should increase LR
                if self.num_good_epochs >= self.patience_increase:
                    self._adjust_lr(self.increase_factor)
                    self.num_good_epochs = 0 # Reset counter after increase
            else:
                # Loss is not improving, reset good epochs and increment bad epochs
                self.num_good_epochs = 0
                self.num_bad_epochs += 1
                
                # Check if we should decrease LR
                if self.num_bad_epochs >= self.patience_decrease:
                    self._adjust_lr(self.factor)
                    self.num_bad_epochs = 0 # Reset counter after decrease
        
        self.last_epoch += 1

    def _adjust_lr(self, factor):
        """Adjusts the learning rate by a given factor for all parameter groups."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = old_lr * factor
            
            # Clamp the new learning rate within the defined bounds
            new_lr = max(self.min_lr, new_lr)
            new_lr = min(self.max_lr, new_lr)
            
            if new_lr != old_lr:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f"Epoch {self.last_epoch}: adjusting learning rate "
                          f"of group {i} from {old_lr:.6f} to {new_lr:.6f}.")