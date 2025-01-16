git submodule update --init --recursive

conda env create -v -f lingo_main_env.yml

conda activate lingo_main

pip install flash-attn --no-build-isolation
