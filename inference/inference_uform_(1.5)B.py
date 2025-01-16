import os
import pandas as pd
import torch
from PIL import Image
from uform.gen_model import VLMForCausalLM, VLMProcessor
from tqdm import tqdm

# Model and processor setup
model = VLMForCausalLM.from_pretrained("unum-cloud/uform-gen").eval().cuda()
processor = VLMProcessor.from_pretrained("unum-cloud/uform-gen")

# Function to make inference with the model
def inference_with_image(image_path, question):
    image = Image.open(image_path)
    prompt = f"[vqa] {question}"  # Using the VQA format for question

    # Preprocess the input text and images
    inputs = processor(texts=[prompt], images=image, return_tensors="pt")
    # Mover los tensores a la GPU
    inputs = {key: value.cuda() for key, value in inputs.items()}
    # Generate the response using the model
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=128,
            eos_token_id=32001,
            pad_token_id=processor.tokenizer.pad_token_id
        )

    # Decode the generated output
    prompt_len = inputs["input_ids"].shape[1]
    decoded_text = processor.batch_decode(output[:, prompt_len:])[0]
    return decoded_text

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
output_csv = 'uform_predictions.csv'  # Output CSV file path
process_and_infer(parquet_file, output_csv)

