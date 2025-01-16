#!/bin/bash
#SBATCH -J LLaVa_Video_Inference # job name
#SBATCH -p gpu # partition name
#SBATCH --gpus 1 # gpu count
#SBATCH --mem-per-gpu 20G # memory per gpu
#SBATCH --cpus-per-task 8 # number of CPUs per task
#SBATCH --mem-per-cpu 4G # memory per CPU
#SBATCH --gres gpu:a100:1 # gpu type

module load cuda
python inference/inference_LLaVa_Video_7B_Qwen2.py val.parquet predictions_llava_video.csv
module unload cuda
