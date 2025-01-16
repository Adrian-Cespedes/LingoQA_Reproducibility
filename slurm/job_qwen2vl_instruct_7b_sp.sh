#!/bin/bash
#SBATCH -J QwenVL_Inference # job name
#SBATCH -p gpu # partition name
#SBATCH --gpus 1 # gpu count
#SBATCH --mem-per-gpu 20G # memory per gpu

module load cuda
python inference/inference_Qwen2VL_Instruct_7B.py val.parquet predictions.csv "You are a Visual Question Answering (VQA) model. Please answer concisely in a maximum of 2 sentences."
module unload cuda
