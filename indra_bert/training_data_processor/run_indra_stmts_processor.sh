#!/bin/bash

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh  # this enables `conda activate` in scripts
conda activate indra_gpt

cd ~/gyorilab/indra_bert/data_processor
python indra_stmts_processor.py \
    --input_path ~/Downloads/indra_benchmark_corpus.json \
    --output_path ../data/indra_benchmark_corpus_annotated.jsonl \
