from typing import List
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
cwd = '/home/rdemello/MSCOCO'

class CustomCocoCaptions(dset.CocoCaptions):
    def _load_target(self, id: int) -> List[str]:
        return super()._load_target(id)[:5] # Some img contain more than 5 captions. Make is so that all has 5 captions.

def get_train_data(batchsize:int):
    train_data = CustomCocoCaptions(root = cwd+'/COCO2014trainimg/train2014',
                        annFile = cwd+'/COCO2014trainValCap/annotations/captions_train2014.json',
                        transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Resize((64,64))]))
    
    train_dl = DataLoader(train_data, batch_size=batchsize, shuffle=True) # shuffle so it won't overfit
    return train_dl

def get_val_data(batchsize: int):
    val_data = CustomCocoCaptions(root = cwd+'/COCO2014valimg/val2014',
                        annFile = cwd+'/COCO2014trainValCap/annotations/captions_val2014.json',
                        transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Resize((64,64))]))
    val_dl = DataLoader(val_data, batch_size=batchsize, shuffle=True) # shuffle so it won't overfit
    return val_dl