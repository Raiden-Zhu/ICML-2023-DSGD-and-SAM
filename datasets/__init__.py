

import time
from .cifar10 import load_cifar10
from .tinyimagenet import load_tinyimagenet

def load_dataset(root, name, image_size, 
                 train_batch_size=64, valid_batch_size=64, 
                 distribute=False, split=None, rank=0, seed=666):
    if name.lower() == 'cifar10':
        return load_cifar10(root=root,  
                            image_size=image_size,  
                            train_batch_size=train_batch_size,
                            valid_batch_size=valid_batch_size,
                            distribute=distribute,
                            split=split,
                            rank=rank,
                            seed=seed
                            ) 
    if name.lower() == 'tinyimagenet':
        return load_tinyimagenet(root=root,  
                            image_size=image_size,  
                            train_batch_size=train_batch_size,
                            valid_batch_size=valid_batch_size,
                            distribute=distribute,
                            split=split,
                            rank=rank,
                            seed=seed
                            ) 
