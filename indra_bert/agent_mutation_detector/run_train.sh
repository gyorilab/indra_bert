#!/bin/bash

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh  # this enables `conda activate` in scripts
conda activate indra_gpt

cd ~/gyorilab/indra_bert

python -m indra_bert.agent_mutation_detector.train \
    --dataset_path data/indra_benchmark_annotated_data/variants_annotated_data/indra_benchmark_corpus_variants_annotated.jsonl \
    --output_dir output/agent_mutation_detection \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --epochs 8 \
    --version 1.0 \
    --use_cached_dataset
