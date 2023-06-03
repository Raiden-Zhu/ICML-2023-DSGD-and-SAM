
import os
import time
import torch
import random
import datetime
import argparse
import numpy as np
import torch.nn as nn

# set random seed
def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if args.device >= 0:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

# get ArgumentParser
def get_args():
    parser = argparse.ArgumentParser()

    ## dataset
    parser.add_argument("--dataset_path", type=str, default='datasets')
    parser.add_argument("--dataset_name", type=str, default='CIFAR10',
                                            choices=['CIFAR10','TinyImageNet'])
    parser.add_argument("--image_size", type=int, default=32, help='input image size')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--n_swap', type=int, default=None)

    # mode parameter
    parser.add_argument('--mode', type=str, default='csgd')
    parser.add_argument('--shuffle', type=str, default="fixed", choices=['fixed', 'random'])
    parser.add_argument('--size', type=int, default=16)
    parser.add_argument('--port', type=int, default=29500)
    parser.add_argument('--backend', type=str, default="gloo")
    # deep model parameter
    parser.add_argument('--model', type=str, default='ResNet18', choices=['ResNet18', 'AlexNet', 'DenseNet'])

    # optimization parameter
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0,  help='weight decay')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--epoch', type=int, default=6000)
    parser.add_argument('--early_stop', type=int, default=6000, help='w.r.t., iterations')
    parser.add_argument('--milestones', type=int, default=[2400, 4800])
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    return args

def add_identity(args, dir_path):
    # set output file name and path, etc. 
    args.identity = (f"{args.dataset_name}s{args.image_size}-"+
                     f"{args.batch_size}-"+
                     f"{args.mode}-"+
                     f"{args.shuffle}-"+
                     f"{args.size}-"+
                     f"{args.model}-"+
                     f"{args.pretrained}-"+
                     f"{args.lr}-"+
                     f"{args.wd}-"+
                     f"{args.gamma}-"+  
                     f"{args.momentum}-"+ 
                     f"{args.warmup_step}-"+ 
                     f"{args.epoch}-"+      
                     f"{args.early_stop}-"+ 
                     f"{args.seed}-"+
                     f"{args.amp}"
                     )       
    args.logs_perf_dir = os.path.join(dir_path,'logs_perf')
    if not os.path.exists(args.logs_perf_dir):
        os.mkdir(args.logs_perf_dir)   
    args.perf_data_dir = os.path.join(args.logs_perf_dir, args.dataset_name)
    if not os.path.exists(args.perf_data_dir):
        os.mkdir(args.perf_data_dir)  
    args.perf_xlsx_dir = os.path.join(args.perf_data_dir, 'xlsx')
    args.perf_imgs_dir = os.path.join(args.perf_data_dir, 'imgs')
    args.perf_dict_dir = os.path.join(args.perf_data_dir, 'dict')
    args.perf_best_dir = os.path.join(args.perf_data_dir, 'best') 

    args.logs_runs_dir = os.path.join(dir_path,'logs_runs')
    if not os.path.exists(args.logs_runs_dir):
        os.mkdir(args.logs_runs_dir)   
    args.runs_data_dir = os.path.join(args.logs_runs_dir, args.dataset_name)
    if not os.path.exists(args.runs_data_dir):
        os.mkdir(args.runs_data_dir) 

    return args



def eval_vision(model, train_loader, valid_loader, epoch, iteration, tb, device):
    criterion=nn.CrossEntropyLoss()
    model.eval()

    print(f"\r")
    total_loss, total_correct, total, step = 0, 0, 0, 0
    start = datetime.datetime.now()
    for batch in train_loader:
        step += 1
        data, target = batch[0].to(device), batch[1].to(device)
        output = model(data)
        p = torch.softmax(output, dim=1).argmax(1)
        total_correct += p.eq(target).sum().item()
        total += len(target)
        loss = criterion(output, target)
        total_loss += loss.item()
        end = datetime.datetime.now()
        print(f"\r"+f"| Evaluate Train | step: {step}, time: {(end - start).seconds}s", flush=True, end="")
    total_train_loss = total_loss / step
    total_train_acc = total_correct / total

    print(f"\r")
    total_loss, total_correct, total, step = 0, 0, 0, 0
    for batch in valid_loader:
        step += 1
        data, target = batch[0].to(device), batch[1].to(device)
        output = model(data)
        p = torch.softmax(output, dim=1).argmax(1)
        total_correct += p.eq(target).sum().item()
        total += len(target)
        loss = criterion(output, target)
        total_loss += loss.item()
        end = datetime.datetime.now()
        print(f"\r| Evaluate Valid | step: {step}, time: {(end - start).seconds}s", flush=True, end="")
    total_valid_loss = total_loss / step
    total_valid_acc = total_correct / total

    if epoch is None:
        tb.add_scalar("valid loss - train loss", total_valid_loss - total_train_loss, iteration)
        tb.add_scalar("valid loss", total_valid_loss, iteration)
        tb.add_scalar("train loss", total_train_loss, iteration)
        tb.add_scalar("valid acc", total_valid_acc, iteration)
        tb.add_scalar("train acc", total_train_acc, iteration)
    else:
        tb.add_scalar("valid loss - train loss", total_valid_loss - total_train_loss, epoch)
        tb.add_scalar("valid loss", total_valid_loss, epoch)
        tb.add_scalar("train loss", total_train_loss, epoch)
        tb.add_scalar("valid acc", total_valid_acc, epoch)
        tb.add_scalar("train acc", total_train_acc, epoch)

    return total_train_acc, total_train_loss, total_valid_acc, total_valid_loss


def eval_vision_amp(model, train_loader, valid_loader, epoch, iteration, tb, device):
    criterion=nn.CrossEntropyLoss()
    model.eval()

    print(f"\r")
    total_loss, total_correct, total, step = 0, 0, 0, 0
    start = datetime.datetime.now()
    for batch in train_loader:
        step += 1
        data, target = batch[0].to(device), batch[1].to(device)
        with torch.cuda.amp.autocast(enabled=True,dtype=torch.float16):
            output = model(data)    
            p = torch.softmax(output, dim=1).argmax(1)
            total_correct += p.eq(target).sum().item()
            total += len(target)
            loss = criterion(output, target)
            total_loss += loss.item()
        end = datetime.datetime.now()
        print(f"\r"+f"| Evaluate Train | step: {step}, time: {(end - start).seconds}s", flush=True, end="")
    total_train_loss = total_loss / step
    total_train_acc = total_correct / total

    print(f"\r")
    total_loss, total_correct, total, step = 0, 0, 0, 0
    for batch in valid_loader:
        step += 1
        data, target = batch[0].to(device), batch[1].to(device)
        with torch.cuda.amp.autocast(enabled=True,dtype=torch.float16):
            output = model(data)
            p = torch.softmax(output, dim=1).argmax(1)
            total_correct += p.eq(target).sum().item()
            total += len(target)
            loss = criterion(output, target)
            total_loss += loss.item()
        end = datetime.datetime.now()
        print(f"\r| Evaluate Valid | step: {step}, time: {(end - start).seconds}s", flush=True, end="")
    total_valid_loss = total_loss / step
    total_valid_acc = total_correct / total

    if epoch is None:
        tb.add_scalar("valid loss - train loss", total_valid_loss - total_train_loss, iteration)
        tb.add_scalar("valid loss", total_valid_loss, iteration)
        tb.add_scalar("train loss", total_train_loss, iteration)
        tb.add_scalar("valid acc", total_valid_acc, iteration)
        tb.add_scalar("train acc", total_train_acc, iteration)
    else:
        tb.add_scalar("valid loss - train loss", total_valid_loss - total_train_loss, epoch)
        tb.add_scalar("valid loss", total_valid_loss, epoch)
        tb.add_scalar("train loss", total_train_loss, epoch)
        tb.add_scalar("valid acc", total_valid_acc, epoch)
        tb.add_scalar("train acc", total_train_acc, epoch)

    return total_train_acc, total_train_loss, total_valid_acc, total_valid_loss


def generate_P(mode, size):
    result = torch.zeros((size, size))
    if mode == "all":
        result = torch.ones((size, size)) / size
    elif mode == "single":
        for i in range(size):
            result[i][i] = 1
    elif mode == "ring":
        for i in range(size):
            result[i][i] = 1 / 3
            result[i][(i - 1 + size) % size] = 1 / 3
            result[i][(i + 1) % size] = 1 / 3
    elif mode == "right":
        for i in range(size):
            result[i][i] = 1 / 2
            result[i][(i + 1) % size] = 1 / 2
    elif mode == "star":
        for i in range(size):
            result[i][i] = 1 - 1 / size
            result[0][i] = 1 / size
            result[i][0] = 1 / size
    elif mode == "meshgrid":
        assert size > 0
        i = int(np.sqrt(size))
        while size % i != 0:
            i -= 1
        shape = (i, size // i)
        nrow, ncol = shape
        print(shape, flush=True)
        topo = np.zeros((size, size))
        for i in range(size):
            topo[i][i] = 1.0
            if (i + 1) % ncol != 0:
                topo[i][i + 1] = 1.0
                topo[i + 1][i] = 1.0
            if i + ncol < size:
                topo[i][i + ncol] = 1.0
                topo[i + ncol][i] = 1.0
        topo_neighbor_with_self = [np.nonzero(topo[i])[0] for i in range(size)]
        for i in range(size):
            for j in topo_neighbor_with_self[i]:
                if i != j:
                    topo[i][j] = 1.0 / max(len(topo_neighbor_with_self[i]),
                                           len(topo_neighbor_with_self[j]))
            topo[i][i] = 2.0 - topo[i].sum()
        result = torch.tensor(topo, dtype=torch.float)
    elif mode == "exponential":
        x = np.array([1.0 if i & (i - 1) == 0 else 0 for i in range(size)])
        x /= x.sum()
        topo = np.empty((size, size))
        for i in range(size):
            topo[i] = np.roll(x, i)
        result = torch.tensor(topo, dtype=torch.float)
    print(result, flush=True)
    return result


def PermutationMatrix(size):
    IdentityMatrix = np.identity(size)
    Permutation = list(range(size))
    np.random.shuffle(Permutation)
    PermutedMatrix = np.take(IdentityMatrix, Permutation, axis=0)
    
    return PermutedMatrix
