import os
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

# Model and tokenizer setup
path = 'OpenGVLab/InternVL2_5-8B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# Helper function to build transform
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

# Load images and preprocess them
def load_images(image_folder, input_size=448):
    transform = build_transform(input_size)
    images = []
    for i in range(5):  # 5 images per segment (0.jpg to 4.jpg)
        img_path = os.path.join(image_folder, f"{i}.jpg")
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0).cuda()  # Add batch dimension
        images.append(img)
    return torch.cat(images, dim=0)  # Concatenate the images for the batch

# Function to make inference with the model
def inference_with_images(image_folder, question, input_size=448):
    pixel_values = load_images(image_folder, input_size).to(torch.bfloat16)

    system_prompt = "You are a Visual Question Answering (VQA) model. Please answer concisely in a maximum of 2 sentences."
    full_question = f"{system_prompt}\n<image>\n{question}"

    generation_config = dict(max_new_tokens=1024, do_sample=True)
    # response, _ = model.chat(tokenizer, pixel_values, question, generation_config)
    response = model.chat(tokenizer, pixel_values, full_question, generation_config)
    
    return response

# Function to process the dataframe and make inferences
def process_and_infer(parquet_file, output_csv, input_size=448):
    df = pd.read_parquet(parquet_file)  # Read the parquet file
    
    # Create a dictionary to hold the results (grouped by question_id)
    results = []
    
    # Get unique question_ids
    question_ids = df['question_id'].unique()
    # question_ids = df['question_id'].unique()[:10]
    
    # Initialize the tqdm progress bar for the question_ids
    for question_id in tqdm(question_ids, desc="Processing questions", unit="question"):
        # Get all segments for this question_id
        question_data = df[df['question_id'] == question_id]
        
        # The first segment_id (we can use it as representative)
        segment_id = question_data.iloc[0]['segment_id']
        question = question_data.iloc[0]['question']
        images_folder = os.path.join('images/val', str(segment_id))  # Folder path for images
        
        # Inference with images and question
        answer = inference_with_images(images_folder, question, input_size)
        
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
