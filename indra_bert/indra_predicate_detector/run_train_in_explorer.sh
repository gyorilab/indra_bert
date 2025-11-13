#!/bin/bash
#SBATCH --job-name=indra_predicate_detector
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=indra_predicate_detector_%j.out
#SBATCH --error=indra_predicate_detector_%j.err

module load cuda/12.3.0
module load anaconda3/2024.06

cd /home/hy.lim/indra_bert
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true

eval "$(conda shell.bash hook)"
conda activate indra

CACHE_DIR=output/predicate_detector/cached_dataset
if [[ "${REBUILD_CACHE:-0}" == "1" ]]; then
  rm -rf "${CACHE_DIR}"
fi

python -m indra_bert.indra_predicate_detector.train \
    --dataset_path data/train/event_trigger_data/event_trigger_dataset_combined_simplified.tsv \
    --output_dir output/predicate_detector \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --epochs 10 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --save_total_limit 1 \
    --use_cached_dataset

python - <<'PY'
from indra_bert.indra_predicate_detector import PredicateDetector

detector = PredicateDetector('output/predicate_detector')
example = '<e>EGF</e> activates EGFR resulting in increased ERK signaling.'
print('\nSample inference:')
result = detector.predict(example)
print(f"Triggers: {result['triggers']}")
print(f"Annotated: {result['annotated_text']}")
PY

python - <<'PY'
from huggingface_hub import upload_folder

repo_id = "thomaslim6793/indra_bert_predicate_detector"
folder_path = "output/predicate_detector"
ignore_patterns = ["checkpoint-*", "checkpoints", "checkpoints/*", "cached_dataset", "cached_dataset/*", "logs", "logs/*"]

print("\nUploading trained model to Hugging Face Hub...")
upload_folder(
    repo_id=repo_id,
    folder_path=folder_path,
    commit_message="Update INDRA predicate detector",
    ignore_patterns=ignore_patterns,
)
print("Upload complete.")
PY

