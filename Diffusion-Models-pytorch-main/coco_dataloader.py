from typing import List
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import json
from typing import Any, Callable, List, Optional, Tuple
from pycocotools.coco import COCO


# cwd = '/mnt/c/Users/rdeme/Documents/Brown/CSCI_2470_Deep_Learning/project/data/''
cwd = '/home/rdemello/MSCOCO/'

class CustomCocoCaptions(dset.CocoCaptions):
    def __init__(self, root: str, annFile1: str, annFile2: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None):
        super().__init__(root, annFile1, transform=transform, target_transform=target_transform, transforms=transforms)
        self.coco2 = COCO(annFile2)
        
    def _load_target(self, id: int):
        super()._load_target(id)
        # Access both image and text embedding ONLY first caption 
        # (image embedding contains list of same image caption for our modified dataset)
        return self.coco.loadAnns(self.coco.getAnnIds(id))[0], self.coco2.loadAnns(self.coco.getAnnIds(id))[0]

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        super().__getitem__(index)
        id = self.ids[index]
        image = self._load_image(id)
        # target 1 represents the image embedding and target 2 represents the text embeddings
        target1, target2 = self._load_target(id)

        if self.transforms is not None:
            image_out, target1 = self.transforms(image, target1)
            _, target2 = self.transforms(image, target2)
        
        # uses cat to combine lists of image and text captions --> target_out [1, 1024]
        target_out = torch.cat([torch.FloatTensor(target1['caption']), torch.FloatTensor(target2['caption'])], dim=0)

        return image_out, target_out
    
def get_train_data(batchsize:int):
    # each set of data has an img and caption
    train_data = CustomCocoCaptions(root = cwd+'COCO2014trainimg/train2014', # img
                        annFile1 = cwd+'COCO2014trainValCap/annotations/train_clip_image.json', # img embedding
                        annFile2 = cwd+'COCO2014trainValCap/annotations/train_clip_text.json', # text embedding
                        transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Resize((80,80))]))
    
    train_dl = DataLoader(train_data, batch_size=batchsize, shuffle=True)
    return train_dl


def get_val_data(batchsize: int):
    val_data = CustomCocoCaptions(root = cwd+'COCO2014valimg/val2014',
                        annFile1 = cwd+'COCO2014trainValCap/annotations/val_clip_image.json',
                        annFile2 = cwd+'COCO2014trainValCap/annotations/val_clip_text.json',
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