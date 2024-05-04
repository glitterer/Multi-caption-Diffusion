from Embedding import clip_text_embedding, clip_image_embedding, t5_embedding
# from coco_dataloader import get_train_data, get_val_data, load_annotations
from cifar_dataloader import get_train_data, get_val_data
import matplotlib.pyplot as plt
import json
import tqdm
from tqdm import tqdm
import torch
from fastprogress import progress_bar
from PIL import Image
import matplotlib.pyplot as plt
from utils import create_embed_json

# data = load_annotations(True)
# ann = data.get('annotations')
# img = data.get('images')
# all_images = {}

# train_path = '/mnt/c/Users/rdeme/Documents/Brown/CSCI_2470_Deep_Learning/project/data/cifar10/cifar10/train'
# val_path = '/mnt/c/Users/rdeme/Documents/Brown/CSCI_2470_Deep_Learning/project/data/cifar10/cifar10/test'

# create_embed_json(train_path, 'train_cifar')
# create_embed_json(val_path, 'val_cifar')

# batch = []
# batch_size = 3000
# max = len(img)
# cur_batch = []
# batch_cap = []
# i=0
# for cur in tqdm(img):
#     i += 1
#     cur_batch.append(cur)
#     image_path = train_path + cur['file_name']
#     img = Image.open(image_path)
#     batch_cap.append(img)
#     if len(cur_batch) == batch_size or i == max:
#         embeded = clip_image_embedding(batch_cap)
#         for j in range(len(cur_batch)):
#             all_images[cur_batch[j]['id']] = embeded[j].tolist()
        
#         cur_batch.clear()
#         batch_cap.clear()


# new_ann = []

# for cur in tqdm(ann):
#     cur['caption'] = all_images[cur['image_id']]
#     new_ann.append(cur)

# data['annotations'] = new_ann

# with open("train_clip_image.json", "w") as outfile:
#     json.dump(data, outfile)

# # Val
# data = load_annotations(False)
# ann = data.get('annotations')
# img = data.get('images')
# all_images = {}
# batch = []
# batch_size = 3000
# max = len(img)
# cur_batch = []
# batch_cap = []
# i=0
# for cur in tqdm(img):
#     i += 1
#     cur_batch.append(cur)
#     image_path = val_path + cur['file_name']
#     img = Image.open(image_path)
#     batch_cap.append(img)
#     if len(cur_batch) == batch_size or i == max:
#         embeded = clip_image_embedding(batch_cap)
#         for j in range(len(cur_batch)):
#             all_images[cur_batch[j]['id']] = embeded[j].tolist()
        
#         cur_batch.clear()
#         batch_cap.clear()


# new_ann = []

# for cur in tqdm(ann):
#     cur['caption'] = all_images[cur['image_id']]
#     new_ann.append(cur)

# data['annotations'] = new_ann

# with open("val_clip_image.json", "w") as outfile:
#     json.dump(data, outfile)
  
  
# for cur in tqdm(ann):
#     i += 1
#     # cur['caption'] = clip_text_embedding([cur.get('caption')]).squeeze().tolist()
#     # new_ann.append(cur)
#     cur_batch.append(cur)
#     batch_cap.append(cur['caption'])
#     if len(cur_batch) == batch_size or i == max:
#         embeded = clip_text_embedding(batch_cap)
#         for j in range(len(cur_batch)):
#             cap = embeded[j].tolist()
#             if len(cap) == 0:
#                 print("Zero detected")
#                 print(batch_cap[j])
#             batch = cur_batch[j]
#             batch['caption'] = cap
#             new_ann.append(batch)
#         cur_batch.clear()
#         batch_cap.clear()
        
     

# data['annotations'] = new_ann

# with open("train_clip_text.json", "w") as outfile:
#     json.dump(data, outfile)
    
# data = load_annotations(False)
# ann = data.get('annotations')

# new_ann = []
# batch = []
# batch_size = 4000
# cur_batch = []
# batch_cap = []
# max = len(ann)
# print(max)
# i = 0
# for cur in tqdm(ann):
#     i += 1
#     # cur['caption'] = clip_text_embedding([cur.get('caption')]).squeeze().tolist()
#     # new_ann.append(cur)
#     cur_batch.append(cur)
#     batch_cap.append(cur['caption'])
#     if len(cur_batch) == batch_size or i == max:
#         embeded = clip_text_embedding(batch_cap)
#         for j in range(len(cur_batch)):
#             cap = embeded[j].tolist()
#             if len(cap) == 0:
#                 print("Zero detected")
#                 print(batch_cap[j])
#             batch = cur_batch[j]
#             batch['caption'] = cap
#             new_ann.append(batch)
#         cur_batch.clear()
#         batch_cap.clear()
        

# data['annotations'] = new_ann

# with open("val_clip_text.json", "w") as outfile:
#     json.dump(data, outfile)

train = get_train_data(1)
pbar = progress_bar(train)
        
for i, (images, ret_class, caption) in enumerate(pbar):
    print(ret_class)
    print(caption)
    images = images.reshape((3,32,32))
    plt.imshow(  images.permute(1, 2, 0)  )
    plt.show()
    
    
#     image = image.permute((0, 2, 3, 1))
#     plt.imshow(image[0].numpy())
#     plt.show()
#     if len(cap) != 5:
#         print(len(cap), "IMPOSTER")
#         print(cap)
    # print(t5_1.shape)
    
# trans = CLIP_embed()
# T5_em = T5_embed()
# out1 = trans(cap[0])
# t5_1 = T5_em(cap[0])
# print(cap[0])
# print(out1)
# print(t5_1)

# out2 = trans(cap[2])
# t5_2 = T5_em(cap[2])
# print(cap[2])
# print(out2)
# print(t5_2.shape)
# out3 = trans(cap[1])
# print(cap[1])
# print(out3)