#!/bin/bash

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh  # this enables `conda activate` in scripts
conda activate indra_gpt

cd ~/gyorilab/indra_bert

python -m ner_agent_detection.train \
    --train_data data/ner_data/BioRED/processed/train_biored_sent_annotated.json \
    --val_data data/ner_data/BioRED/processed/val_biored_sent_annotated.json \
    --test_data data/ner_data/BioRED/processed/test_biored_sent_annotated.json \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --output_dir output/ner_agent_detection
