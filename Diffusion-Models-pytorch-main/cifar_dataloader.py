from typing import List
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import json
from typing import Any, Callable, List, Optional, Tuple
from pycocotools.coco import COCO
from PIL import Image



cwd = '/mnt/c/Users/rdeme/Documents/Brown/CSCI_2470_Deep_Learning/project/data/cifar10/cifar10/'
# cwd = '/home/rdemello/cifar10/cifar10/'

class cifar_dataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        """
        Arguments:
            json_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(json_file) as annot_json:
            self.json = json.load(annot_json)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.json)

    def __getitem__(self, index) -> tuple[Any, Any, Any]:
        # super().__getitem__(index)
        file = self.json[index]
        path = self.root_dir + '/' + file['class'] + '/' + file['file_name']
        cifar_image = Image.open(path)
        if self.transform is not None:
            cifar_image = self.transform(cifar_image)
        ret_class = -1
        if file['class'] == 'airplane':
            ret_class = 0
        elif file['class'] == 'automobile':
            ret_class = 1
        elif file['class'] == 'bird':
            ret_class = 2
        elif file['class'] == 'cat':
            ret_class = 3
        elif file['class'] == 'deer':
            ret_class = 4
        elif file['class'] == 'dog':
            ret_class = 5
        elif file['class'] == 'frog':
            ret_class = 6
        elif file['class'] == 'horse':
            ret_class = 7
        elif file['class'] == 'ship':
            ret_class = 8
        elif file['class'] == 'truck':
            ret_class = 9
        caption = torch.Tensor(file['caption'])
        return cifar_image, ret_class, caption

def get_train_data(batchsize:int):
    train_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = cifar_dataset(cwd + 'train_cifar.json',cwd + 'train', transform=train_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    
    return train_dataloader

def get_val_data(batchsize:int):
    val_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    val_dataset = cifar_dataset(cwd + 'val_cifar.json',cwd + 'test', transform=val_transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True)
    return val_dataloader