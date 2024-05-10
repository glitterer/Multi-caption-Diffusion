import clip
import torch
from transformers import AutoTokenizer, T5EncoderModel
from coco_dataloader import get_train_data, get_val_data

'''
Source CLIP Model: https://github.com/openai/CLIP?tab=readme-ov-file#zero-shot-prediction
This will take a caption, tokenize the caption then use the a pretrained CLIP Model to encode the tokenized caption data.

params: 'caption'- A list of descriptive sentences for an input image.
return: 'text_encoding' - Returns a list of text encodings of the inputed caption
'''

def clip_text_embedding(caption):

    # print("CLIP TEXT EMBEDING")
    # Load the pretrained CLIP model & ensure your device is using CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load('ViT-B/32', device)

    # Prepare the text inputs by tokenizing the sentence
    # print("Original Sentences:")
    # [print(c) for c in caption]
    text_inputs = torch.cat([clip.tokenize(c) for c in caption]).to(device)
    
    # Calculate features
    # https://github.com/openai/CLIP?tab=readme-ov-file#modelencode_imageimage-tensor
    # model.encode_text(text: Tensor): Given a batch of text tokens, returns the text features encoded by the language portion of the CLIP model.
    with torch.no_grad():
        # Uses model to create text embedding
        text_embedding = model.encode_text(text_inputs)
    
    # Print Shapes of encodings
    # print("Text Embedding:", text_embedding[0].shape)

    return text_embedding

'''
Source CLIP Model: https://github.com/openai/CLIP?tab=readme-ov-file#zero-shot-prediction
This will take a PIL image, preprocess the image from a PIL to a Tensor then encode the unsqueezed Tensor image.

params: 'image'- A PIL image.
return: 'image_embedding' - Returns an image encoding based on a pretrained CLIP model of the inputed image
'''
def clip_image_embedding(images):

    # print("CLIP IMAGE EMBEDING")

    # Load the pretrained CLIP model & ensure your device is using CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # Prepare the input image to go from a PIL to a Tensor image and unsqueeze it
    all_img = []
    for image in images:
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        all_img.append(image_input)
    all_img = torch.stack(all_img)
    all_img = all_img.squeeze()
    # print("Image size", image_input.shape)

    # Calculate features
    # https://github.com/openai/CLIP?tab=readme-ov-file#modelencode_imageimage-tensor
    # model.encode_image(image: Tensor): Given a batch of images, returns the image features encoded by the vision portion of the CLIP model.
    with torch.no_grad():
        # Uses model to create image embedding
        image_embedding = model.encode_image(all_img)

    # Print Shapes of encodings
    # print("Image Embedding", image_embedding.shape)

    return image_embedding

'''
Source T5 Model: https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.TFT5EncoderModel
This will take a caption, tokenize the caption then use the a pretrained T5 Model to encode the tokenized caption data.

params: 'caption'- A list of descriptive sentences for an input image.
return: 'text_embedding' - Returns a list of text encodings of the inputed captions
'''
def t5_embedding(caption):

    # print("T5 TEXT EMBEDING")

    # Load the pretrained T5 model and T5 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    model = T5EncoderModel.from_pretrained("google-t5/t5-small")
    
    # Create a list to store the Tensor caption embedding
    text_embedding = []

    # Loop through the list of captions
    for c in caption:
        # Tokenize the caption sentece
        input_ids = tokenizer(c, return_tensors='pt').input_ids

        # Run it through the pretrained T5 model, then access the last hidden state to get the Tensor representation of the output of the T5 encoder 
        last_hidden_states = model(input_ids=input_ids).last_hidden_state
        text_embedding.append(last_hidden_states)

    return text_embedding[0]
