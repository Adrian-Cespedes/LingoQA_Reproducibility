import os
import pandas as pd
import torch
import argparse
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from tqdm import tqdm
import copy
import numpy as np

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Visual Question Answering Inference with LLava-Video-7B-Qwen2")
    parser.add_argument("parquet_file", type=str, help="Path to the input parquet file")
    parser.add_argument("output_csv", type=str, help="Path to the output CSV file")
    parser.add_argument("system_prompt", type=str, default="", nargs="?", help="Optional system prompt for the model")

    return parser.parse_args()

# Load the LLava-Video model and tokenizer
pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)

model.eval()

# Function to load and preprocess the video frames
def load_frames(image_folder, num_frames=5):
    frames = []
    for i in range(num_frames):
        frame_path = os.path.join(image_folder, f"{i}.jpg")
        if os.path.exists(frame_path):
            frames.append(Image.open(frame_path))
        else:
            print(f"Warning: Frame {frame_path} does not exist.")
            frames.append(Image.new("RGB", (336, 336)))  # Create a blank image if a frame is missing
    frames = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].cuda().half()
    return frames

# Function to process the frames and make inferences with LLava
def inference_with_frames(image_folder, question, system_prompt=""):
    # Load frames and preprocess them
    frames = load_frames(image_folder)

    # Create the conversation template for LLava
    conv_template = "qwen_1_5"  # Use appropriate template
    question_text = DEFAULT_IMAGE_TOKEN + f"{system_prompt}\n{question}" if system_prompt else question
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question_text)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    # Tokenize the input and generate a response
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    response = model.generate(input_ids, images=frames, modalities=["video"], do_sample=False, temperature=0, max_new_tokens=4096)

    output_text = tokenizer.batch_decode(response, skip_special_tokens=True)[0].strip()
    return output_text

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
        image_folder = os.path.join('images', str(segment_id))  # Folder path for images

        # Inference with frames and question
        answer = inference_with_frames(image_folder, question, system_prompt)

        results.append({
            'question_id': question_id,
            'segment_id': segment_id,
            'answer': answer
        })

    # Save results to CSV
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
