from Embedding import clip_text_embedding, clip_image_embedding, t5_embedding
from coco_dataloader import get_train_data, get_val_data, load_annotations
import matplotlib.pyplot as plt
import json
import tqdm
from tqdm import tqdm

data = load_annotations()
ann = data.get('annotations')

new_ann = []
image_ids = []
batch = []
batch_size = 488
cur_batch = []
batch_cap = []

for cur in tqdm(ann):
    if cur['image_id'] not in image_ids:
        # cur['caption'] = clip_text_embedding([cur.get('caption')]).squeeze().tolist()
        # new_ann.append(cur)
        image_ids.append(cur['image_id'])
        cur_batch.append(cur)
        batch_cap.append(cur['caption'])
        if len(cur_batch) == batch_size:
            embeded = clip_text_embedding(batch_cap)
            for i, batch in enumerate(cur_batch):
                cap = embeded[0]
                batch['caption'] = cap.tolist()
                new_ann.append(batch)
            cur_batch.clear()
            batch_cap.clear()
            
     

data['annotations'] = new_ann
with open("test.json", "w") as outfile:
    json.dump(data, outfile)

# train = get_train_data(1)
# for image, cap in train:
#     print(cap[0])
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