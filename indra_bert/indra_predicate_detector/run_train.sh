#!/bin/bash

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate indra

cd ~/gyorilab/indra_bert

python -m indra_bert.indra_predicate_detector.train \
    --dataset_path data/train/event_trigger_data/event_trigger_dataset_combined_simplified_sample_1000.tsv \
    --output_dir output/predicate_detector \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --epochs 3 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --use_cached_dataset

