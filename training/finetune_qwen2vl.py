from PIL import Image
from unsloth import FastVisionModel # FastLanguageModel for LLMs
from datasets import Dataset, load_dataset, concatenate_datasets, interleave_datasets, load_from_disk
import torch
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import shutil
from tqdm import tqdm
import os
import random


BASE_FOLDER = "training"


# dataset = load_dataset("parquet", data_files={'train': 'train.parquet', 'test': 'test.parquet'})
scenery_dataset = load_dataset("parquet", data_files='training/scenery/train.parquet', streaming=True)
action_dataset = load_dataset("parquet", data_files='training/action/train.parquet', streaming=True)

# dataset = concatenate_datasets([scenery_dataset["train"], action_dataset["train"]])
# combined_dataset = interleave_datasets([scenery_dataset["train"], action_dataset["train"]])

# print("Dataset concatenated")

# Function to process each dataset
def process_and_convert_dataset(dataset, dataset_type):
    converted_data = []
    count = sum(1 for _ in dataset)
    with tqdm(total=count, desc=f"Converting {dataset_type} samples") as pbar:
        for sample in dataset:
            converted = process_sample(sample, dataset_type)
            if converted:
                converted_data.append(converted)
            pbar.update(1)
    return converted_data

def process_sample(sample, dataset_type):
    """Process individual sample and convert to VLM-friendly format"""
    try:
        # Process images
        img_paths = [os.path.join(BASE_FOLDER, dataset_type, img) for img in sample["images"][:5]]
        # processed_images = [Image.open(p).convert("RGB").resize((448, 448)) for p in img_paths]

        # Build conversation structure
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        *[{
                            "type": "image",
                            "image": image,
                            "max_pixels": 448*448,
                        } for image in img_paths],
                        {"type": "text", "text": sample["question"]}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": sample["answer"]}
                    ]
                }
            ]
        }
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None


converted_scenery = process_and_convert_dataset(scenery_dataset["train"], "scenery")
print("Scenery dataset processed.")

converted_action = process_and_convert_dataset(action_dataset["train"], "action")
print("Action dataset processed.")

combined_dataset = converted_scenery + converted_action

random.shuffle(combined_dataset)

print("Dataset loaded successfully.")

# converted_dataset = [sample['messages'] for sample in tqdm(converted_dataset, desc="Processing messages")]

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-7B-Instruct",
    load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    local_files_only = False, # Use local files only
    low_cpu_mem_usage = True,
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
    local_files_only = True, # Use local files only,
    low_cpu_mem_usage = True,
)

print("Model loaded successfully.")


FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = combined_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 1,
        warmup_steps = 4000,
        max_steps = 40000,
        # num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate = 5e-5,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.1,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",     # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
    ),
)

print("Training model...")
trainer_stats = trainer.train()
print("Training complete!")

# dump trainer stats to txt
with open("training_stats.txt", "w") as f:
    f.write(trainer_stats)

model.save_pretrained_merged("Qwen2VL_Instruct_7B_LingoFinetune", tokenizer,)
print("Model saved to Qwen2VL_Instruct_7B_LingoFinetune.")
