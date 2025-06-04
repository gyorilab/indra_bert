#!/bin/bash

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate indra_gpt

cd ~/gyorilab/indra_bert

python -m indra_bert.indra_stmt_classifier.train \
    --dataset data/indra_benchmark_annotated_data/indra_benchmark_corpus_annotated_stratified_sample.jsonl \
    --model_name SpanBERT/spanbert-base-cased \
    --output_dir output/indra_stmt_classifier
