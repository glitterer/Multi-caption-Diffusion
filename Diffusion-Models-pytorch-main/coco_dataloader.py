from typing import List
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import json
cwd = '/home/rdemello/MSCOCO/'

class CustomCocoCaptions(dset.CocoCaptions):
    def _load_target(self, id: int):
        captions = super()._load_target(id)
        captions = torch.Tensor(captions).type(torch.float32)
        return captions[0]

def get_train_data(batchsize:int):
    train_data = CustomCocoCaptions(root = cwd+'COCO2014trainimg/train2014',
                        annFile = cwd+'COCO2014trainValCap/annotations/train_clip_text.json',
                        transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Resize((80,80))]))
    
    train_dl = DataLoader(train_data, batch_size=batchsize, shuffle=True)
    return train_dl

def get_val_data(batchsize: int):
    val_data = CustomCocoCaptions(root = cwd+'COCO2014valimg/val2014',
                        annFile = cwd+'COCO2014trainValCap/annotations/val_clip_text.json',
                        transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Resize((80,80))]))
    val_dl = DataLoader(val_data, batch_size=batchsize, shuffle=True)
    return val_dl

def load_annotations(train_not_val = True):
    if train_not_val == True:
        path = cwd+'/annotations/captions_train2014.json'
    else:
        path = cwd+'/annotations/captions_val2014.json'
    with open(path) as annot_json:
        data = json.load(annot_json)
        return data
            