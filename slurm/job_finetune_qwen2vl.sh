#!/bin/bash
#SBATCH -J Qwen2VL_Finetune # job name
#SBATCH -p gpu # partition name
#SBATCH --gpus 1 # gpu count
#SBATCH --mem-per-gpu 20G # memory per gpu
#SBATCH --cpus-per-task 8 # number of CPUs per task
#SBATCH --nodelist=ag001

module load cuda
python training/train_Qwen2VL.py
module unload cuda
