#!/bin/bash

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh  # this enables `conda activate` in scripts
conda activate indra_gpt

cd ~/gyorilab/indra_bert

python -m indra_stmt_agents_ner_model.train_indra_stmt_agents_ner \
    --dataset data/indra_benchmark_corpus_annotated_stratified_sample.jsonl \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --output_dir output/indra_stmt_agents_ner
