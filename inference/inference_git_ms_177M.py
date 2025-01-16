import os
import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and processor setup
processor = AutoProcessor.from_pretrained("microsoft/git-base-textvqa")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-textvqa").to(device)


# Function to make inference with the model
def inference_with_image(image_path, question):
    image = Image.open(image_path)
    indice = question.find("?")
    prompt = f"{question[:indice]}?"  # Using the VQA format for question
    #print(prompt)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    # Preprocess the input text and images
    input_ids = processor(text=prompt, add_special_tokens=False).input_ids
    input_ids = [processor.tokenizer.cls_token_id] + input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    
    # Generate the response using the model
    with torch.inference_mode():
        output = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)

    # Decode the generated output
    decoded_text = processor.batch_decode(output, skip_special_tokens=True)
    return decoded_text[0].split("?")[1]

# Function to process the dataframe and make inferences
def process_and_infer(parquet_file, output_csv):
    df = pd.read_parquet(parquet_file)  # Read the parquet file    
    results = []
    
    question_ids = df['question_id'].unique()
    
    for question_id in tqdm(question_ids, desc="Processing questions", unit="question"):
        # Get all segments for this question_id
        question_data = df[df['question_id'] == question_id]
        
        # The first segment_id (we can use it as representative)
        segment_id = question_data.iloc[0]['segment_id']
        question = question_data.iloc[0]['question']
        image_path = os.path.join('images/val', str(segment_id))  # Folder path for images
        
        # Inference with images and question
        answer = inference_with_image(image_path + "/0.jpg", question)
        
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
output_csv = 'git_ms_predictions.csv'  # Output CSV file path
process_and_infer(parquet_file, output_csv)

