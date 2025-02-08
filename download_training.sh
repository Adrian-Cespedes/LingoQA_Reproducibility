#!/bin/bash

# Scenery Train dataset
gdown https://drive.google.com/drive/folders/1GiwWGfrM8pO27CYLu_9Uwtdcz0JoqHr7 -O training/ --folder

# Action Train dataset
gdown https://drive.google.com/drive/folders/1QQqBrR3uGDC05Zc11zMeui6Zzl7RvFZg -O training/ --folder

huggingface-cli download unsloth/Qwen2-VL-7B-Instruct
