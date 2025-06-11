#!/bin/bash

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh  # this enables `conda activate` in scripts
conda activate indra_gpt

cd ~/gyorilab/indra_bert

python -m indra_bert.indra_agent_role_assigner.train \
    --dataset_path data/indra_benchmark_annotated_data/indra_benchmark_corpus_annotated_stratified_sample_2000.jsonl \
    --output_dir output/indra_stmt_agents_role_assigner \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --epochs 8 \
    --version 1.0 \
    --use_cached_dataset
