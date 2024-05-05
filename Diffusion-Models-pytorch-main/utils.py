import os, random
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from os import listdir
from os.path import isfile, join
from Embedding import clip_image_embedding
from tqdm import tqdm
import json



def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try: torch.manual_seed(s)
    except NameError: pass
    try: torch.cuda.manual_seed_all(s)
    except NameError: pass
    try: np.random.seed(s%(2**32-1))
    except NameError: pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def one_batch(dl):
    return next(iter(dl))
        

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def mk_folders(run_name):
    os.makedirs("cifar_2_models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("cifar_models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def create_embed_json(path:str, json_name:str):
    '''
    Takes in the path to Cifar directory, and creates a json including file name, class and clip image embedding
    '''
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    json_file = []
    for cur_class in classes:
        cwd = path + '/' + cur_class
        batch_size = 2000
        batch_fname = []
        batch_img = []
        all_files = len(listdir(cwd))
        i = 0
        j = 0
        for file in tqdm(listdir(cwd)):
            i += 1
            j += 1
            cur_image = Image.open(cwd + '/' + file)
            
            batch_fname.append(file)
            batch_img.append(cur_image)
            if i == batch_size or j == all_files:
                embed_images = clip_image_embedding(batch_img)
                for k in range(len(embed_images)):
                    cur_dict = {}
                    cur_dict['file_name'] = batch_fname[k]
                    cur_dict['class'] = cur_class
                    cur_dict['caption'] = embed_images[k].tolist()
                    json_file.append(cur_dict)
                i = 0
                batch_img.clear()
                batch_fname.clear()
    out_name = json_name + '.json'
    print(len(json_file), "LEngth")
    with open(out_name, "w") as outfile:
        json.dump(json_file, outfile)