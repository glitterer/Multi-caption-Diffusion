#!/bin/bash

#SBATCH --job-name="ddpm_1_epoch_1_batch"
#SBATCH --partition=gpus
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-10:00:00
#SBATCH -e job_error_%x_%j.e
#SBATCH -o job_out_%x_%j.o
#SBATCH --nodes=1
#SBATCH --exclude=gpu[1601-1605],gpu[1701-08],gpu[1801-02],gpu[1901-07],gpu[2002-03]
#SBATCH --gpus-per-node=1

source activate ~/miniconda3/envs/DLproject
cd ~/CSCI2470Project/Multi-caption-Diffusion/Diffusion-Models-pytorch-main

python train.py