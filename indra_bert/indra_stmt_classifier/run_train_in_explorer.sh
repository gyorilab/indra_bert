#!/bin/bash
#SBATCH --job-name=indra_stmt_classifier
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=indra_stmt_classifier_%j.out
#SBATCH --error=indra_stmt_classifier_%j.err

module load cuda/12.3.0
module load anaconda3/2024.06

cd /home/hy.lim/indra_bert
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true

eval "$(conda shell.bash hook)"
conda activate indra

CACHE_DIR=output/indra_stmt_classifier/cached_dataset
if [[ "${REBUILD_CACHE:-0}" == "1" ]]; then
  rm -rf "${CACHE_DIR}"
fi

python -m indra_bert.indra_stmt_classifier.train \
    --relation_path data/train/statement_classification/combined/relation_binary.jsonl \
    --indra_path data/train/indra_benchmark_annotated_data/indra_benchmark_corpus_annotated_stratified_sample_2000.jsonl \
    --output_dir output/indra_stmt_classifier \
    --epochs 10 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 3e-5 \
    --gate1_loss_weight 1.0 \
    --gate2_loss_weight 0.5 \
    --gate3_loss_weight 0.25 \
    --eval_strategy epoch \
    --cache_dir "${CACHE_DIR}" \
    --use_cached_dataset

python - <<'PY'
from indra_bert.indra_stmt_classifier.inference import IndraStmtClassifier

classifier = IndraStmtClassifier('output/indra_stmt_classifier')
example = '<e>BRCA1</e> activates <e>RAD51</e> in DNA repair.'
print('\nSample inference:')
print(classifier.predict(example))
PY
