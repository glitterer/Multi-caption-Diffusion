import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
cwd = '/mnt/c/Users/rdeme/Documents/Brown/CSCI_2470_Deep_Learning/project/data'

def get_train_data():
    train_data = dset.CocoCaptions(root = cwd+'/train2014',
                        annFile = cwd+'/annotations/captions_train2014.json',
                        transform=transforms.PILToTensor())
    return train_data

def get_val_data():
    val_data = dset.CocoCaptions(root = cwd+'/val2014',
                        annFile = cwd+'/annotations/captions_val2014.json',
                        transform=transforms.PILToTensor())
    return val_data