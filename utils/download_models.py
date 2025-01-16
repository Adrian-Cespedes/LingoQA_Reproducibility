from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer

# Model and tokenizer names
model_name = "Qwen/Qwen2-VL-7B-Instruct"

# Download model, tokenizer, and processor
print("Downloading model, tokenizer, and processor...")
Qwen2VLForConditionalGeneration.from_pretrained(model_name)
AutoProcessor.from_pretrained(model_name)
AutoTokenizer.from_pretrained(model_name)

print("Model, tokenizer, and processor downloaded and saved to default location.")
