#!/bin/bash
#SBATCH --job-name=indra_bert_stmt_classifier_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=indra_bert_stmt_classifier_train_%j.out
#SBATCH --error=indra_bert_stmt_classifier_train_%j.err

# Load modules
module load cuda/12.3.0
module load anaconda3/2024.06

# Set up environment
cd /home/hy.lim/indra_bert
export CUDA_VISIBLE_DEVICES=0

# Use conda run instead of activate (avoids conda init issues)
conda run -n indra python -m indra_bert.indra_stmt_classifier.train \
    --dataset_path data/indra_benchmark_annotated_data/indra_benchmark_corpus_annotated_stratified_sample_2000.jsonl \
    --output_dir output/indra_stmt_classifier \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --epochs 8 \
    --version 1.1 \
    --max_negatives_per_positive 5
