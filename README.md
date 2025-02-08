# LingoQA Reproducibility
This repo contains some scripts to reproduce/enhance the results presented in the [LingoQA](https://arxiv.org/abs/2312.14115) paper.

> The scripts are designed to be run on a SLURM cluster. Still, instructions for both SLURM and non-SLURM environments are provided.

## Installation

- If you're using a SLURM cluster, make sure to load the necessary modules before running the scripts. Example: `module load cuda`

To set up the main environment, run:

```bash
chmod +x ./install.sh && ./install.sh
```

Additionally, you may need to install other environments for specific tasks:

- LingoJudge:

```bash
chmod +x ./install_judge.sh && ./install_judge.sh
```

- Unsloth for fine-tuning:

```bash
chmod +x ./install_unsloth.sh && ./install_unsloth.sh
```

### Dataset & Model Downloads
Before running the models, download the required datasets and models:

```bash
conda activate lingo_main
chmod +x ./download_eval_dataset.sh && ./download_eval_dataset.sh
chmod +x ./download_models.sh && ./download_models.sh
```

If you plan to fine-tune models, you also need to download the training dataset:

```bash
chmod +x ./download_training.sh && ./download_training.sh
```

## Usage
### Running Pre-trained Models
If you only want to run inference using pre-trained models, you can use the scripts inside the slurm/ directory. If you're not using a SLURM cluster, you can extract the Python command from these scripts and run it directly.

Example:

```bash
# SLURM
sbatch slurm/job_qwen2vl_instruct_7b.sh
# Non-SLURM
python inference/inference_qwen2vl_instruct_7b.py val.parquet predictions_qwen2vl.csv
```

### Fine-tuning Models
To fine-tune models, first ensure that the training dataset has been downloaded. Then, execute the appropriate training script:

```bash
# SLURM
sbatch slurm/job_finetune_qwen2vl.sh
# Non-SLURM
python training/finetune_qwen2vl.py
```

## Notes
- You may need to adjust `--nodelist` flags in the SLURM scripts to match your cluster's configuration.

## Obtained Results
| Model | Fine-tune | System Prompt | Size | Frame quantity | Accuracy (%) |
|---------------------|-----------|---------------|----------------|------------------|---------------|
| [InternVL 2.5](https://huggingface.co/OpenGVLab/InternVL2_5-8B)         | No        | No            | 8B             | 5                | 49.0          |
| [InternVL 2.5](https://huggingface.co/OpenGVLab/InternVL2_5-8B)         | No        | Yes            | 8B             | 5                | 47.4          |
| [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)             | No        | No            | 7B             | 5                | 50.2          |
| [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)             | No        | Yes            | 7B             | 5                | 52.2          |
| [Uforms](https://huggingface.co/unum-cloud/uform-gen)              | No        | No            | 1.5B  | 1                | 46.4          |
| GIT-base-textvqa    | No        | No            | 177M           | 5                | 31.2          |