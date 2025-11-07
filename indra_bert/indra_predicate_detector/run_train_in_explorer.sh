#!/bin/bash
#SBATCH --job-name=indra_pred_detector_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=indra_pred_detector_train_%j.out
#SBATCH --error=indra_pred_detector_train_%j.err

module load cuda/12.3.0
module load anaconda3/2024.06

cd /home/hy.lim/indra_bert
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=true

eval "$(conda shell.bash hook)"
conda activate indra

python -m indra_bert.indra_predicate_detector.train \
    --train-data data/train/predicate_detection_data/llm_annotated/llm_annotated_benchmark_stratified_filtered.jsonl \
    --output-dir output/predicate_detector \
    --model-name bert-base-uncased \
    --num-epochs 10 \
    --batch-size 16 \
    --val-ratio 0.1 \
    --max-seq-length 256 \
    --learning-rate 5e-5 \
    --debug-examples 3
