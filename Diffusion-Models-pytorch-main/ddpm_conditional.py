"""
This code train a conditional diffusion model on CIFAR.
It is based on @dome272.

@wandbcode{condition_diffusion}
"""

import argparse, logging, copy
from types import SimpleNamespace
from contextlib import nullcontext

import torch
import matplotlib.pyplot as plt
import torchvision
from torch import optim
import torch.nn as nn
import numpy as np
from fastprogress import progress_bar

from Embedding import clip_text_embedding, clip_image_embedding, t5_embedding
from utils import *
from modules import UNet_conditional, EMA, UNet
from coco_dataloader import get_train_data, get_val_data


config = SimpleNamespace(    
    run_name = "uncon_ddpm",
    epochs = 40,
    noise_steps=1000,
    seed = 42,
    batch_size = 8,
    img_size = 80,
    text_embed_length = 256,
    train_folder = "train",
    val_folder = "test",
    device = "cuda",
    slice_size = 1,
    do_validation = True,
    fp16 = True,
    log_every_epoch = 10,
    num_workers=10,
    lr = 5e-3)


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size1=80, img_size2=80, text_embed_length=256, c_in=3, c_out=3, device="cuda", **kwargs):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.cap_reduce = torch.nn.Sequential(torch.nn.Linear(512, 256), torch.nn.LeakyReLU()).to(device)
        self.img_size1 = img_size1
        self.img_size2 = img_size2
        self.model = UNet(c_in, c_out,  **kwargs).to(device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.device = device
        self.c_in = c_in
        self.text_embed_length = text_embed_length
        # self.cap_enc = clip_text_embedding

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        
        Ɛ = torch.randn_like(x)
        noisy_img = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ
        
        return noisy_img, Ɛ
    
    @torch.inference_mode()
    def sample(self, use_ema, labels, cfg_scale=3):
        model = self.ema_model if use_ema else self.model
        n = len(labels)
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.inference_mode():
            labels = self.cap_reduce(labels).to(self.device)
            x = torch.randn((n, self.c_in, self.img_size1, self.img_size2)).to(self.device)
            for i in progress_bar(reversed(range(1, self.noise_steps)), total=self.noise_steps-1, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
    
                
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def train_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        self.scheduler.step()

    def one_epoch(self, train=True):
        avg_loss = 0.
        if train: self.model.train()
        else: self.model.eval()
        pbar = progress_bar(self.train_dataloader)
        if train:
            batches = len(self.train_dataloader)
        else:
            batches = len(self.val_dataloader)
        stop = int(batches/2)
        for i, (images, labels) in enumerate(pbar):
            with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                
                images = images.type(torch.FloatTensor).to(self.device)
                # labels = labels.to(self.device)
                # labels = self.cap_reduce(labels)
                
                t = self.sample_timesteps(images.shape[0]).to(self.device)
                
                x_t, noise = self.noise_images(images, t)
                # if np.random.random() < 0.15:
                labels = None
                predicted_noise = self.model(x_t, t)
                loss = self.mse(noise, predicted_noise)
                avg_loss += loss.cpu().detach()
            if train:
                self.train_step(loss)
                print("train_mse " + str(loss.item()) + " learning_rate "+ str(self.scheduler.get_last_lr()[0]) + " batch:" + str(i) + " of 5174")
            pbar.comment = f"MSE={loss.item():2.3f}"
            
            if i == stop:
                break
        return avg_loss.mean().item()

    # def log_images(self, epoch):
    #     "Log images to save them to disk"
    #     labels1 = self.cap_enc(['A zebra walking on the street']).type(torch.float32).to(self.device)
    #     labels2 = self.cap_enc(['A car on grass']).type(torch.float32).to(self.device)
    #     labels = torch.cat([labels1, labels2])
    #     labels = labels.reshape((2, 512))
    #     sampled_images = self.sample(use_ema=False, labels=labels)
    #     sampled_images = sampled_images.permute((0, 2, 3, 1))
    #     sampled_images = sampled_images.cpu().detach().numpy()
    #     plt.imsave(f'img1_e{epoch}_full.png', sampled_images[0])
    #     plt.imsave(f'img2_e{epoch}_full.png',sampled_images[1])
        

    def load(self, model_cpkt_path, model_ckpt="checkpt_e10.pt", ema_model_ckpt="ema_checkpt_e10.pt"):
        self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt)))
        self.ema_model.load_state_dict(torch.load(os.path.join(model_cpkt_path, ema_model_ckpt)))

    def save_model(self, run_name, epoch=-1):
        "Save model locally"
        torch.save(self.model.state_dict(), os.path.join("models", run_name, f"checkpt_e{epoch}.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join("models", run_name, f"ema_checkpt_e{epoch}.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", run_name, f"optim_e{epoch}.pt"))
        

    def prepare(self, args):
        mk_folders(args.run_name)
        self.train_dataloader = get_train_data(config.batch_size)
        self.val_dataloader = get_val_data(config.batch_size)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()

    def fit(self, args):
        for epoch in progress_bar(range(args.epochs), total=args.epochs, leave=True):
            logging.info(f"Starting epoch {epoch}:")
            print("Epoch ",epoch)
            _  = self.one_epoch(train=True)
            
            ## validation
            if args.do_validation:
                avg_loss = self.one_epoch(train=False)
                print("Val_mse", avg_loss)
            
            
            # self.log_images(epoch)
            self.save_model(run_name=args.run_name, epoch=epoch)

        # save model
        self.save_model(run_name=args.run_name, epoch=epoch)




def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--img_size', type=int, default=config.img_size, help='image size')
    parser.add_argument('--text_embed_length', type=int, default=config.text_embed_length, help='number of classes')
    parser.add_argument('--dataset_path', type=str, default=config.dataset_path, help='path to dataset')
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parser.add_argument('--slice_size', type=int, default=config.slice_size, help='slice size')
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='noise steps')
    args = vars(parser.parse_args())
    
    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)


if __name__ == '__main__':
    parse_args(config)

    ## seed everything
    set_seed(config.seed)

    diffuser = Diffusion(config.noise_steps, img_size=config.img_size, text_embed_length=config.text_embed_length)
    diffuser.prepare(config)
    diffuser.fit(config)
