
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

class Worker_Vision:
    def __init__(self, model, rank, optimizer, scheduler,
                 train_loader, device):       
        self.model = model
        self.rank = rank
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        # self.train_loader_iter = train_loader.__iter__()
        self.device = device


    def update_iter(self):
        self.train_loader_iter = self.train_loader.__iter__()

    def step(self):
        self.model.train()

        batch = self.train_loader_iter.next()
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        output = self.model(data)
        loss = criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()

    def step_csgd(self):
        self.model.train()

        batch = self.train_loader_iter.next()
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        output = self.model(data)
        loss = criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()

        grad_dict = {}
        for name, param in self.model.named_parameters():
            grad_dict[name] = param.grad.data

        return grad_dict

    def update_grad(self):
        self.optimizer.step()
        self.scheduler.step()

    def scheduler_step(self):
        self.scheduler.step()

from torch.cuda.amp.grad_scaler import GradScaler
scaler = GradScaler()
class Worker_Vision_AMP:
    def __init__(self, model, rank, optimizer, scheduler,
                 train_loader, device):       
        self.model = model
        self.rank = rank
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        # self.train_loader_iter = train_loader.__iter__()
        self.device = device


    def update_iter(self):
        self.train_loader_iter = self.train_loader.__iter__()

    def step(self):
        self.model.train()

        batch = self.train_loader_iter.next()
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        with torch.cuda.amp.autocast(enabled=True,dtype=torch.float16):
            output = self.model(data)
            loss = criterion(output, target)
        self.optimizer.zero_grad()
        scaler.scale(loss).backward()
        # loss.backward()

    def step_csgd(self):
        self.model.train()

        batch = self.train_loader_iter.next()
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        with torch.cuda.amp.autocast(enabled=True,dtype=torch.float16):
            output = self.model(data)
            loss = criterion(output, target)
        self.optimizer.zero_grad()
        scaler.scale(loss).backward()
        # loss.backward()

        grad_dict = {}
        for name, param in self.model.named_parameters():
            grad_dict[name] = param.grad.data

        return grad_dict

    def update_grad(self):
        # self.optimizer.step()
        scaler.step(self.optimizer)
        scaler.update()
        self.scheduler.step()

    def scheduler_step(self):
        self.scheduler.step()
