
# CIFAR10 training 50000
# 50000/(512*16) = 6.10
# 50000/(64*16)  = 48.8

## AlexNet
python main.py --dataset_name "CIFAR10" --image_size 64 --batch_size 64 --mode "csgd" --size 16 --lr 0.1 --model "AlexNet_M" --warmup_step 0 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
python main.py --dataset_name "CIFAR10" --image_size 64 --batch_size 512 --mode "csgd" --size 16 --lr 0.8 --model "AlexNet_M" --warmup_step 0 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0


## ResNet18
python main.py --dataset_name "CIFAR10" --image_size 56 --batch_size 64 --mode "csgd" --size 16 --lr 0.1 --model "ResNet18_M" --warmup_step 60 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
python main.py --dataset_name "CIFAR10" --image_size 56 --batch_size 512 --mode "csgd" --size 16 --lr 0.8 --model "ResNet18_M" --warmup_step 60 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
# ResNet18 + LAMB (Layer-wise Adaptive Moments optimizer for Batching training (LAMB))
python main_lamb.py --dataset_name "CIFAR10" --image_size 56 --batch_size 64 --mode "csgd" --size 16 --lr 0.007 --model "ResNet18_M" --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
python main_lamb.py --dataset_name "CIFAR10" --image_size 56 --batch_size 512 --mode "csgd" --size 16 --lr 0.02 --model "ResNet18_M" --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
# without pretrain
python main.py --dataset_name "CIFAR10" --image_size 56 --batch_size 64 --mode "csgd" --size 16 --lr 0.1 --model "ResNet18_M" --milestones 4800 9600 --early_stop 12000 --epoch 12000 --seed 666 --pretrained 0 --device 0


## ResNet34
python main.py --dataset_name "CIFAR10" --image_size 56 --batch_size 64 --mode "csgd" --size 16 --lr 0.1 --model "ResNet34_M" --warmup_step 60 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
python main.py --dataset_name "CIFAR10" --image_size 56 --batch_size 512 --mode "csgd" --size 16 --lr 0.8 --model "ResNet34_M" --warmup_step 60 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0


## DenseNet121
# training with amp（automatic mixed precision）
python main.py --dataset_name "CIFAR10" --image_size 56 --batch_size 64 --mode "csgd" --size 16 --lr 0.1 --model "DenseNet121_M" --warmup_step 150 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0 --amp
python main.py --dataset_name "CIFAR10" --image_size 56 --batch_size 512 --mode "csgd" --size 16 --lr 0.8 --model "DenseNet121_M" --warmup_step 150 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0 --amp
