#!/bin/bash
#SBATCH -J LLaVa-Video - Inference # job name
#SBATCH -p gpu # partition name
#SBATCH --gpus 1 # gpu count
#SBATCH --mem-per-gpu 20G # memory per gpu

module load cuda
python inference/inference_LLaVa_Video_7B_Qwen2.py val.parquet predictions_llava_video.csv
module unload cuda
