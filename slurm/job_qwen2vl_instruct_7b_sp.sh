#!/bin/bash
#SBATCH -J QwenVL_SP_Inference # job name
#SBATCH -p gpu # partition name
#SBATCH --gpus 1 # gpu count
#SBATCH --mem-per-gpu 20G # memory per gpu
#SBATCH --cpus-per-task 8 # number of CPUs per task
#SBATCH --nodelist=ag001

module load cuda
python inference/inference_Qwen2VL_Instruct_7B.py val.parquet predictions_qwen2vl_sp.csv "You are a Visual Question Answering (VQA) model. Please answer concisely in a maximum of 2 sentences."
module unload cuda
