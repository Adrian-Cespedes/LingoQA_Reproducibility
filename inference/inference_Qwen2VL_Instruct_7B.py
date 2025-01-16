import os
import pandas as pd
import torch
import argparse
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Visual Question Answering Inference")
    parser.add_argument("parquet_file", type=str, help="Path to the input parquet file")
    parser.add_argument("output_csv", type=str, help="Path to the output CSV file")
    parser.add_argument("system_prompt", type=str, default="", help="Optional system prompt for the model")

    return parser.parse_args()

# Load the model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Function to make inference with the model
def inference_with_images(image_folder, question, system_prompt=""):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": [
                        f"file://{os.path.join(image_folder, '0.jpg')}",
                        f"file://{os.path.join(image_folder, '1.jpg')}",
                        f"file://{os.path.join(image_folder, '2.jpg')}",
                        f"file://{os.path.join(image_folder, '3.jpg')}",
                        f"file://{os.path.join(image_folder, '4.jpg')}",
                    ],
                    "fps": 1.0,
                },
                # Add the system prompt if provided, otherwise just use the question
                {"type": "text", "text": f"{system_prompt}\n{question}" if system_prompt else question},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

# Function to process the dataframe and make inferences
def process_and_infer(parquet_file, output_csv, system_prompt=""):
    df = pd.read_parquet(parquet_file)

    # Create a dictionary to hold the results (grouped by question_id)
    results = []

    # Get unique question_ids
    question_ids = df['question_id'].unique()

    # Initialize the tqdm progress bar for the question_ids
    for question_id in tqdm(question_ids, desc="Processing questions", unit="question"):
        # Get all segments for this question_id
        question_data = df[df['question_id'] == question_id]

        segment_id = question_data.iloc[0]['segment_id']
        question = question_data.iloc[0]['question']
        images_folder = os.path.join('images/val', str(segment_id))  # Folder path for images

        # Inference with images and question
        answer = inference_with_images(images_folder, question, system_prompt)

        results.append({
            'question_id': question_id,
            'segment_id': segment_id,
            'answer': answer
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

def main():
    args = parse_args()

    print("Arguments:")
    print(f"Input parquet file: {args.parquet_file}")
    print(f"Output CSV file: {args.output_csv}")
    print(f"System prompt: {args.system_prompt}")

    print("Starting inference...")

    process_and_infer(args.parquet_file, args.output_csv, args.system_prompt)

if __name__ == "__main__":
    main()
