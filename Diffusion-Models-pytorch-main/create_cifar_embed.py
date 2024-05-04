from os import listdir
from os.path import isfile, join
from Embedding import clip_image_embedding
import torch

def create_embed_json(path:str, json_name:str):
    '''
    Takes in the path to Cifar directory, and creates a json including file name, class and clip image embedding
    '''
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for cur_class in classes:
        cwd = path + '/' + cur_class
        print(cwd)
        for file in listdir(cwd):
            print(file)