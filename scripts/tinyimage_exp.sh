


## ResNet18
python main.py --dataset_name "TinyImageNet" --image_size 56 --batch_size 64 --mode "exponential" --shuffle 'random' --size 16 --lr 0.1 --model "ResNet18_M" --warmup_step 120 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
python main.py --dataset_name "TinyImageNet" --image_size 56 --batch_size 512 --mode "exponential" --shuffle 'random' --size 16 --lr 0.8 --model "ResNet18_M" --warmup_step 120 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0

