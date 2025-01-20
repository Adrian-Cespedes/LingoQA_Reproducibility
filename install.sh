git submodule update --init --recursive

conda env create -v -f lingo_main_env.yml

conda activate lingo_main

pip install flash-attn --no-build-isolation

pip install -U "huggingface_hub[cli]"
huggingface-cli download google/siglip-so400m-patch14-384
huggingface-cli download google/siglip-so400m-patch14-384
huggingface-cli download google/siglip-so400m-patch14-384
