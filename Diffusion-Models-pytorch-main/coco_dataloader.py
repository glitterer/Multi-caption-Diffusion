import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
cwd = '/mnt/c/Users/rdeme/Documents/Brown/CSCI_2470_Deep_Learning/project/data'

def get_train_data(batchsize:int):
    train_data = dset.CocoCaptions(root = cwd+'/train2014',
                        annFile = cwd+'/annotations/captions_train2014.json',
                        transform=transforms.PILToTensor())
    
    
    return [train_data[0], train_data[1], train_data[2], train_data[3], train_data[4]]

def get_val_data(batchsize: int):
    val_data = dset.CocoCaptions(root = cwd+'/val2014',
                        annFile = cwd+'/annotations/captions_val2014.json',
                        transform=transforms.PILToTensor())
    val_dl = DataLoader(val_data, batch_size=batchsize, shuffle=False)
    return val_dl