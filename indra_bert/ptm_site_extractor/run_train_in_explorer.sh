#!/bin/bash
#SBATCH --job-name=indra_bert_ptm_site_extractor_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=indra_bert_ptm_site_extractor_train_%j.out
#SBATCH --error=indra_bert_ptm_site_extractor_train_%j.err

# Load modules
module load cuda/12.3.0
module load anaconda3/2024.06

# Set up environment
cd /home/hy.lim/indra_bert
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=true

# Activate conda
eval "$(conda shell.bash hook)"
conda activate indra

python -m indra_bert.ptm_site_extractor.train \
    --dataset_path data/train/ptm_data/sentence_level_training_dedup.tsv \
    --output_dir output/ptm_site_extractor \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --epochs 10 \
    --version 1.0 \
    --batch_size 16 \
    --resume_from_checkpoint latest \
    --use_cached_dataset

