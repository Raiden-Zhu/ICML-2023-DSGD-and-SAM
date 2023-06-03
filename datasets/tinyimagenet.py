from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import pickle
from torch.utils.data import DataLoader
import torchvision.transforms as tfs
from .distribute_dataset import distribute_dataset


class TinyImageNet(Dataset):
    def __init__(self, root='./data', train=True, transform=None):
        root = os.path.join(root, 'tiny-imagenet')
        if train:
            root = os.path.join(root, 'tiny-imagenet_train.pkl')
        else:
            root = os.path.join(root, 'tiny-imagenet_val.pkl')
        with open(root, 'rb') as f:
            dat = pickle.load(f)
        self.data = dat['data']
        self.targets = dat['targets']
        self.transform = transform

    def __getitem__(self, item):
        data, targets = Image.fromarray(self.data[item]), self.targets[item]
        if self.transform is not None:
            data = self.transform(data)
        return data, targets

    def __len__(self):
        return len(self.data)


def load_tinyimagenet(root, transforms=None, image_size=32,  
                    train_batch_size=64, valid_batch_size=64,
                    distribute=False, split=1.0, rank=0, seed=666):
    if transforms is None:
        transforms = tfs.Compose([
            tfs.Resize((image_size,image_size)),
            tfs.ToTensor(),
            tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    if train_batch_size is None:
        train_batch_size = 1
    if split is None:
        split = [1.0]
    train_set = TinyImageNet(root, True, transforms)
    valid_set = TinyImageNet(root, False, transforms)
    if distribute:
        train_set = distribute_dataset(train_set, split, rank, seed=seed)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=valid_batch_size, drop_last=True)
    return train_loader, valid_loader, (3, image_size, image_size), 200

