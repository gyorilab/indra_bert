#!/bin/bash
#SBATCH --job-name=indra_bert_agent_mutation_detector_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=indra_bert_agent_mutation_detector_train_%j.out
#SBATCH --error=indra_bert_agent_mutation_detector_train_%j.err

# Load modules
module load cuda/12.3.0
module load anaconda3/2024.06

# Set up environment
cd /home/hy.lim/indra_bert
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true

# Activate conda
eval "$(conda shell.bash hook)"
conda activate indra

python -m indra_bert.agent_mutation_detector.train \
    --dataset_path data/pubtator3/pubtator3_BioCXML_0_protein_variants.json \
    --output_dir output/agent_mutation_detection \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --epochs 15 \
    --version 2.0 \
    --pubtator3_format \
    --use_cached_dataset \
    --max_negative_examples_per_agent 1 \
    --max_total_examples 100000

