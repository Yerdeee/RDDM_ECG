import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import math

class CosineAnnealingLRWarmup(lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs  # 선형 증가
            else:
                cosine_decay = 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
                return cosine_decay * (1 - min_lr) + min_lr  # Cosine Annealing

        super().__init__(optimizer, lr_lambda, last_epoch)
