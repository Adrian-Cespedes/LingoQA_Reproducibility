from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from llava.model.builder import load_pretrained_model

# Model and tokenizer names
model_name = "Qwen/Qwen2-VL-7B-Instruct"

# Download model, tokenizer, and processor
print("Downloading model, tokenizer, and processor...")
Qwen2VLForConditionalGeneration.from_pretrained(model_name)
AutoProcessor.from_pretrained(model_name)
AutoTokenizer.from_pretrained(model_name)

print("Model, tokenizer, and processor downloaded and saved to default location.")


# Load the LLava-Video model and tokenizer
pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)

print("LLava-Video model, tokenizer, and image processor loaded successfully.")
