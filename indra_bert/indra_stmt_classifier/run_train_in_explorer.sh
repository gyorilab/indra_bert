#!/bin/bash
#SBATCH --job-name=indra_stmt_classifier
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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
    --indra_path data/train/indra_benchmark_annotated_data/indra_benchmark_corpus_annotated_stratified_sample_2000_with_heuristic_filtered_negatives.jsonl \
    --hrt_path data/train/event_trigger_data/event_trigger_dataset_combined_hrt.tsv \
    --output_dir output/indra_stmt_classifier \
    --epochs 10 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 3e-5 \
    --gate1_loss_weight 1.0 \
    --gate2_loss_weight 1.0 \
    --gate3_loss_weight 1.0 \
    --gate4_loss_weight 1.0 \
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

python - <<'PY'
from huggingface_hub import upload_folder

repo_id = "thomaslim6793/indra_bert_indra_stmt_classifier"
folder_path = "output/indra_stmt_classifier"
ignore_patterns = ["checkpoint-*", "cached_dataset", "cached_dataset/*"]

print("\nUploading trained model to Hugging Face Hub...")
upload_folder(
    repo_id=repo_id,
    folder_path=folder_path,
    commit_message="Update multitask INDRA statement classifier",
    ignore_patterns=ignore_patterns,
)
print("Upload complete.")
PY
