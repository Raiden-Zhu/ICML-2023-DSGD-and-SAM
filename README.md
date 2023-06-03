# [ICML 2023] Decentralize to Generalize? Decentralized SGD and Average-direction SAM are Asymptotically Equivalent

The repository contains the offical implementation of the paper

> [**ICML 2023**] **Decentralized SGD and Average-direction SAM are Asymptotically Equivalent**

## Example of usage

#### Train ResNet-18 on CIFAR-10 using D-SGD and C-SGD with 1024 total batch sizes.

```
python main.py --dataset_name "CIFAR10" --image_size 56 --batch_size 64 --mode "csgd" --size 16 --lr 0.1 --model "ResNet18_M" --warmup_step 60 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0

python main.py --dataset_name "CIFAR10" --image_size 56 --batch_size 64 --mode "ring" --size 16 --lr 0.1 --model "ResNet18_M" --warmup_step 60 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
```

#### Train ResNet-18 on CIFAR-10 using D-SGD and C-SGD with 8192 total batch sizes.

```
python main.py --dataset_name "CIFAR10" --image_size 56 --batch_size 512 --mode "csgd" --size 16 --lr 0.8 --model "ResNet18_M" --warmup_step 60 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0

python main.py --dataset_name "CIFAR10" --image_size 56 --batch_size 512 --mode "ring" --size 16 --lr 0.8 --model "ResNet18_M" --warmup_step 60 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
```

More detailed scripts can be found in the "scripts" folder.
