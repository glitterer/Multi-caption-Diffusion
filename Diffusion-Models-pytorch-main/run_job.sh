#!/bin/bash

#SBATCH --job-name="ddpm_1_epoch_1_batch"
#SBATCH --partition=gpus
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-10:00:00
#SBATCH -e job_error_%x_%j.e
#SBATCH -o job_out_%x_%j.o
#SBATCH --nodes=1
#SBATCH --exclude=gpu[1601-1605,1701-1708,1801-1802,1901-1904,1905-1906,2002-2003]
#SBATCH --gpus-per-node=1

source activate ~/miniconda3/envs/DLproject
cd ~/Multi-caption-Diffusion/Diffusion-Models-pytorch-main
nvidia-smi
python train.py