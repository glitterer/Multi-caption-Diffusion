from Embedding import clip_text_embedding, clip_image_embedding, t5_embedding
from coco_dataloader import get_train_data, get_val_data

train = get_train_data(1)
for image, cap in train:
    if len(cap) != 5:
        print(len(cap), "IMPOSTER")
        print(cap)
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