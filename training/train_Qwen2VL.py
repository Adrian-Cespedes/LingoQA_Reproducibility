from unsloth import FastVisionModel # FastLanguageModel for LLMs
from datasets import load_dataset, concatenate_datasets
import torch
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

BASE_FOLDER = "training"

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-7B-Instruct",
    load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    local_files_only = True, # Use local files only
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
)

print("Model loaded successfully.")

# dataset = load_dataset("parquet", data_files={'train': 'train.parquet', 'test': 'test.parquet'})
scenery_dataset = load_dataset("parquet", data_files='training/scenery/train.parquet')
action_dataset = load_dataset("parquet", data_files='training/action/train.parquet')

def concatenate_path(sample, dataset_type):
    return [f"{BASE_FOLDER}/{dataset_type}/{img_path}" for img_path in sample["images"]]

scenery_dataset = scenery_dataset.map(concatenate_path, with_indices=True)
action_dataset = action_dataset.map(concatenate_path, with_indices=True)
dataset = concatenate_datasets([scenery_dataset, action_dataset])

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {
                "type": "video",
                  "video": [
                      f"file://{sample["image"][0]}",
                      f"file://{sample["image"][1]}",
                      f"file://{sample["image"][2]}",
                      f"file://{sample["image"][3]}",
                      f"file://{sample["image"][4]}",
                  ],
                  "fps": 1.0,
            },
            {"type" : "text",  "text"  : sample["question"]},
          ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["answer"]} ]
        },
    ]
    return { "messages" : conversation }

converted_dataset = dataset.map(convert_to_conversation)

print("Dataset loaded successfully.")

FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 1,
        warmup_steps = 1000,
        max_steps = 10000,
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
