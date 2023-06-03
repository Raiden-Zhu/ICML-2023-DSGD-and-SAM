

import os
import copy
import torch
import socket
import datetime
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from datasets import load_dataset
from networks import load_model
from workers.worker_vision import *
from utils.scheduler import Warmup_MultiStepLR
from utils.utils import *
from lamb.pytorch_lamb import Lamb, log_lamb_rs

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
dir_path = os.path.dirname(__file__)

nfs_dataset_path1 = '/mnt/nfs4-p1/ckx/datasets/'
nfs_dataset_path2 = '/nfs4-p1/ckx/datasets/'

# torch.set_num_threads(4) 

def main(args):
    set_seed(args)

    # check nfs dataset path
    if os.path.exists(nfs_dataset_path1):
        args.dataset_path = nfs_dataset_path1
    elif os.path.exists(nfs_dataset_path2):
        args.dataset_path = nfs_dataset_path2

    log_id = datetime.datetime.now().strftime('%b%d_%H:%M:%S') + '_' + socket.gethostname() + '_' + args.identity
    writer = SummaryWriter(log_dir=os.path.join(args.runs_data_dir, log_id))

    probe_train_loader, probe_valid_loader, _, classes = load_dataset(root=args.dataset_path, name=args.dataset_name, image_size=args.image_size,
                                                                    train_batch_size=256, valid_batch_size=64)
    worker_list = []
    split = [1.0 / args.size for _ in range(args.size)]
    for rank in range(args.size):
        train_loader, _, _, classes = load_dataset(root=args.dataset_path, name=args.dataset_name, image_size=args.image_size, 
                                                    train_batch_size=args.batch_size, 
                                                    distribute=True, rank=rank, split=split, seed=args.seed)
        model = load_model(args.model, classes, pretrained=args.pretrained).to(args.device)

        optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(.9, .999), adam=('lamb' == 'adam'))

        scheduler = MultiStepLR(optimizer, milestones=[max(args.epoch, args.early_stop), max(args.epoch, args.early_stop)], gamma=args.gamma)

        if args.amp:
            worker = Worker_Vision_AMP(model, rank, optimizer, scheduler, train_loader, args.device)
        else:
            worker = Worker_Vision(model, rank, optimizer, scheduler, train_loader, args.device)
        worker_list.append(worker)


    # 定义 中心模型 center_model
    center_model = copy.deepcopy(worker_list[0].model)
    # center_model = copy.deepcopy(worker_list[0].model)
    for name, param in center_model.named_parameters():
        for worker in worker_list[1:]:
            param.data += worker.model.state_dict()[name].data
        param.data /= args.size

    P = generate_P(args.mode, args.size)
    iteration = 0
    for epoch in range(args.epoch):  
        for worker in worker_list:
            worker.update_iter()   
        for _ in range(train_loader.__len__()):
            if args.mode == 'csgd':
                for worker in worker_list:
                    worker.model.load_state_dict(center_model.state_dict())
                    worker.step()
                    worker.update_grad()
            else: # dsgd
                # 每个iteration，传播矩阵P中的worker做random shuffle（自己的邻居在下一个iteration时改变）
                if args.shuffle == "random":
                    P_perturbed = np.matmul(np.matmul(PermutationMatrix(args.size).T,P),PermutationMatrix(args.size)) 
                elif args.shuffle == "fixed":
                    P_perturbed = P
                model_dict_list = []
                for worker in worker_list:
                    model_dict_list.append(worker.model.state_dict())  
                for worker in worker_list:
                    worker.step()
                    for name, param in worker.model.named_parameters():
                        param.data = torch.zeros_like(param.data)
                        for i in range(args.size):
                            p = P_perturbed[worker.rank][i]
                            param.data += model_dict_list[i][name].data * p
                    # worker.step() # 效果会变差
                    worker.update_grad()

            center_model = copy.deepcopy(worker_list[0].model)
            for name, param in center_model.named_parameters():
                for worker in worker_list[1:]:
                    param.data += worker.model.state_dict()[name].data
                param.data /= args.size
            
            if iteration % 50 == 0:    
                start_time = datetime.datetime.now() 
                eval_iteration = iteration
                if args.amp:
                    train_acc, train_loss, valid_acc, valid_loss = eval_vision_amp(center_model, probe_train_loader, probe_valid_loader,
                                                                                None, iteration, writer, args.device)                    
                else:
                    train_acc, train_loss, valid_acc, valid_loss = eval_vision(center_model, probe_train_loader, probe_valid_loader,
                                                                                None, iteration, writer, args.device)
                print(f"\n|\033[0;31m Iteration:{iteration}|{args.early_stop}, epoch: {epoch}|{args.epoch},\033[0m",
                        f'train loss:{train_loss:.4}, acc:{train_acc:.4%}, '
                        f'valid loss:{valid_loss:.4}, acc:{valid_acc:.4%}.',
                        flush=True, end="\n")
            else:
                end_time = datetime.datetime.now() 
                print(f"\r|\033[0;31m Iteration:{eval_iteration}-{iteration}, time: {(end_time - start_time).seconds}s\033[0m", flush=True, end="")
            iteration += 1
            if iteration == args.early_stop: break
        if iteration == args.early_stop: break

    state = {
        'acc': train_acc,
        'epoch': epoch,
        'state_dict': center_model.state_dict() 
    }    
    if not os.path.exists(args.perf_dict_dir):
        os.mkdir(args.perf_dict_dir)  
    torch.save(state, os.path.join(args.perf_dict_dir, log_id + '.t7'))

    writer.close()        
    print('ending')

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    ## dataset
    parser.add_argument("--dataset_path", type=str, default='datasets')
    parser.add_argument("--dataset_name", type=str, default='CIFAR10',
                                            choices=['CIFAR10','CIFAR100','TinyImageNet'])
    parser.add_argument("--image_size", type=int, default=56, help='input image size')
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument('--n_swap', type=int, default=None)

    # mode parameter
    parser.add_argument('--mode', type=str, default='ring', choices=['csgd', 'ring', 'meshgrid', 'exponential'])
    parser.add_argument('--shuffle', type=str, default="fixed", choices=['fixed', 'random'])
    parser.add_argument('--size', type=int, default=16)
    parser.add_argument('--port', type=int, default=29500)
    parser.add_argument('--backend', type=str, default="gloo")
    # deep model parameter
    parser.add_argument('--model', type=str, default='ResNet18', 
                        choices=['ResNet18', 'AlexNet', 'DenseNet121', 'AlexNet_M','ResNet18_M', 'ResNet34_M', 'DenseNet121_M'])
    parser.add_argument("--pretrained", type=int, default=1)

    # optimization parameter
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0,  help='weight decay')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--warmup_step', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=6000)
    parser.add_argument('--early_stop', type=int, default=6000, help='w.r.t., iterations')
    parser.add_argument('--milestones', type=int, nargs='+', default=[2400, 4800])
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--amp", action='store_true', help='automatic mixed precision')
    args = parser.parse_args()

    args = add_identity(args, dir_path)
    # print(args)
    main(args)
