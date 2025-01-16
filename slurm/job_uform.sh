#!/bin/bash
#SBATCH -J uform # job name
#SBATCH -p gpu # partition name
#SBATCH --gpus 1 # gpu count
#SBATCH --mem-per-gpu 20G # memory per gpu

module load cuda
python inference/inference_uform_1-5B.py
module unload cuda