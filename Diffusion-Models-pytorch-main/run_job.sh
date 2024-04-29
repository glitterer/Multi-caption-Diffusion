#!/bin/bash

#SBATCH --job-name="ddpm_1_epoch_1_batch"
#SBATCH --partition=gpus
#SBATCH --mem=250G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-10:00:00
#SBATCH -e job_error_%x_%j.e
#SBATCH -o job_out_%x_%j.o
#SBATCH --nodes=1
#SBATCH --exclude=gpu[1601-1605]

source activate ~/miniconda3/envs/DLproject
cd ~/CSCI2470Project/Multi-caption-Diffusion/Diffusion-Models-pytorch-main

python train.py