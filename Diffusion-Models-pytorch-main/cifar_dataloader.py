from typing import List
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import json
from typing import Any, Callable, List, Optional, Tuple
from pycocotools.coco import COCO


cwd = '/mnt/c/Users/rdeme/Documents/Brown/CSCI_2470_Deep_Learning/project/data/cifar10/cifar10/'
# cwd = '/home/rdemello/MSCOCO/'

def get_train_data(batchsize:int):
    train_transforms = transforms.Compose([
        transforms.Resize(32 + int(.25*32)),  # args.img_size + 1/4 *args.img_size
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = dset.ImageFolder(cwd + 'train', transform=train_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    
    return train_dataloader

def get_val_data(batchsize:int):
    val_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    val_dataset = dset.ImageFolder(cwd + 'test', transform=val_transforms)
    val_dataset = DataLoader(val_dataset, batch_size=batchsize, shuffle=True)