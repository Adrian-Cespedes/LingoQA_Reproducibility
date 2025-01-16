import os
import pandas as pd
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# default: Load the model on the available device(s)
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# Function to make inference with the model
def inference_with_images(image_folder, question):
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
                {"type": "text", "text": question},
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
def process_and_infer(parquet_file, output_csv):
    df = pd.read_parquet(parquet_file)  # Read the parquet file

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
        answer = inference_with_images(images_folder, question)

        # Store results for both entries (one for each segment_id)
        results.append({
            'question_id': question_id,
            'segment_id': segment_id,  # This is per segment
            'answer': answer  # Same answer for both segments (since it's the same question)
        })

    # Convert results to DataFrame and export to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

# Example usage
parquet_file = 'val.parquet'  # Path to your parquet file
output_csv = 'predictions.csv'  # Output CSV file path
process_and_infer(parquet_file, output_csv)
