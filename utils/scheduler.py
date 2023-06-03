
import math
from torch.optim.lr_scheduler import LambdaLR

class Warmup_MultiStepLR(LambdaLR):
    def __init__(self, optimizer, warmup_step=10, milestones=[20, 30], gamma=0.1, init_rate=0.0, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_step = warmup_step
        self.milestones = milestones
        self.milestones_rate = 1.0
        self.gamma = gamma
        self.init_rate = init_rate
        self.last_epoch = last_epoch
        super(Warmup_MultiStepLR, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_warmup(self, step):
        # linear warmup
        lr_rate = (float(step) / float(max(1, self.warmup_step)))*(1-self.init_rate) + self.init_rate
        # # sine warm up
        # progress = float(step) / float(max(1, self.warmup_step))
        # lr_rate = math.sin(math.pi * float(0.5) * progress)
        # return lr_rate
        
        if lr_rate<=0.1:
            return 0.1
        else:
            return lr_rate

    def lr_decay(self, step):
        if step in self.milestones:
            self.milestones_rate = self.milestones_rate*self.gamma
        return self.milestones_rate

    def lr_lambda(self, step):
        if step < self.warmup_step:
            return self.lr_warmup(step)
        return self.lr_decay(step)
        


