#!/bin/bash

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate indra_gpt

cd ~/gyorilab/indra_bert

python -m indra_bert.indra_stmt_classifier.train \
    --dataset_path data/indra_benchmark_annotated_data/indra_benchmark_corpus_annotated_stratified_sample.jsonl \
    --output_dir output/indra_stmt_classifier \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --epochs 8 \
    --version 1.0 \
    --max_negatives_per_positive 5
