import torch
from torchvision.utils import save_image
from ddpm import Diffusion
from coco_dataloader import get_val_data
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.batch_size = 1  # 5
args.image_size = 64
args.dataset_path = r"/mnt/c/Users/rdeme/Documents/Brown/CSCI_2470_Deep_Learning/project/data/temp"
torch.manual_seed(1)
dataloader = get_val_data(1)

diff = Diffusion(device="cpu")

image = next(iter(dataloader))[0]
t = torch.Tensor([50, 100, 150, 200, 300, 600, 700, 999]).long()

noised_image, _ = diff.noise_images(image, t)
save_image(image, "orig.jpg")
save_image(noised_image.add(1).mul(0.5), "noise.jpg")
