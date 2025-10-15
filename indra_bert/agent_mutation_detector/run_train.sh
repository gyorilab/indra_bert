#!/bin/bash

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh  # this enables `conda activate` in scripts
conda activate indra_gpt

cd ~/gyorilab/indra_bert

python -m indra_bert.agent_mutation_detector.train \
    --dataset_path data/pubtator3/pubtator3_BioCXML_0_protein_variants.json \
    --output_dir output/agent_mutation_detection \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --epochs 15 \
    --version 2.0 \
    --pubtator3_format \
    --use_cached_dataset
