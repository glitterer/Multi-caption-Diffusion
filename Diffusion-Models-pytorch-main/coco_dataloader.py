from typing import List
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import json
cwd = '/mnt/c/Users/rdeme/Documents/Brown/CSCI_2470_Deep_Learning/project/data'

class CustomCocoCaptions(dset.CocoCaptions):
    def _load_target(self, id: int) -> List[str]:
        return super()._load_target(id)[:5]

def get_train_data(batchsize:int):
    train_data = CustomCocoCaptions(root = cwd+'/train2014',
                        annFile = cwd+'/annotations/captions_train2014.json',
                        transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Resize((80,80))]))
    
    train_dl = DataLoader(train_data, batch_size=batchsize, shuffle=True)
    return train_dl

def get_val_data(batchsize: int):
    val_data = CustomCocoCaptions(root = cwd+'/val2014',
                        annFile = cwd+'/annotations/captions_val2014.json',
                        transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Resize((84,84))]))
    val_dl = DataLoader(val_data, batch_size=batchsize, shuffle=True)
    return val_dl

def load_annotations():
    path = cwd+'/annotations/captions_val2014.json'
    with open(path) as annot_json:
        data = json.load(annot_json)
        return data
            