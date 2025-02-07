git submodule update --init --recursive

conda create --name lingo_main \
    python=3.10.9 \
    pip \
    -y

conda activate lingo_main
conda config --add channels conda-forge

pip install -r lingo_main_requirements.txt
pip install flash-attn --no-build-isolation
pip install -U "huggingface_hub[cli]"
