#!/bin/bash

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh  # this enables `conda activate` in scripts
conda activate indra_gpt

cd ~/gyorilab/indra_bert/ner_model
python train_ner.py \
    --dataset ../data/indra_benchmark_corpus_annotated_small_sample.jsonl \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --output_dir ../output/ner_model
