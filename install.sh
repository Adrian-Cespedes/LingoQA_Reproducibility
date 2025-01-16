git submodule update --init --recursive

conda env create -vv -f lingo_main_env.yml

conda activate lingo_main

pip install flash-attn --no-build-isolation
