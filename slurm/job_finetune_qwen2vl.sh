#!/bin/bash
#SBATCH -J Qwen2VL_Finetune # job name
#SBATCH -p gpu # partition name
#SBATCH --gpus 1 # gpu count
#SBATCH --mem-per-gpu 120G # memory per gpu
#SBATCH --cpus-per-task 32 # number of CPUs per task
#SBATCH --nodelist=g003

module load cuda gnu12 autotools
python training/train_Qwen2VL.py
module unload cuda gnu12 autotools
