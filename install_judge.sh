#!/bin/bash

conda create --name lingo_judge \
    python=3.10.9 \
    pip \
    -y

conda activate lingo_main
conda config --add channels conda-forge

pip install -r LingoQA/requirements.txt
