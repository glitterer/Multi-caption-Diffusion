#!/bin/bash

#SBATCH --job-name="ddpm_uncon"
#SBATCH --partition=gpus
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --time=4-10:00:00
#SBATCH -e job_%x_%j.e
#SBATCH -o job_%x_%j.o
#SBATCH --nodes=1
#SBATCH --exclude=gpu[1601-1605,1701-1708,1801-1802,1901-1904,1905-1906,2002-2003]
#SBATCH --gpus-per-node=2

source activate ~/miniconda3/envs/DLproject
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
cd ~/CSCI2470Project/Multi-caption-Diffusion/Diffusion-Models-pytorch-main
nvidia-smi
python train.py