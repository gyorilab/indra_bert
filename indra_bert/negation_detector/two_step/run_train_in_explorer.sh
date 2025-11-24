#!/bin/bash
#SBATCH --job-name=negation_detector_two_step
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=negation_detector_two_step_%j.out
#SBATCH --error=negation_detector_two_step_%j.err

module load cuda/12.3.0
module load anaconda3/2024.06

cd /home/hy.lim/indra_bert
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true

eval "$(conda shell.bash hook)"
conda activate indra

CACHE_DIR=output/negation_detector/cached_dataset
if [[ "${REBUILD_CACHE:-0}" == "1" ]]; then
  rm -rf "${CACHE_DIR}"
fi

python -m indra_bert.negation_detector.two_step.train \
    --data_path data/train/negation_data/combined_negation_dataset.json \
    --output_dir output/negation_detector \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --epochs 10 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --train_split 0.8 \
    --val_split 0.1 \
    --eval_strategy epoch \
    --cache_dir "${CACHE_DIR}" \
    --use_cached_dataset

python - <<'PY'
from indra_bert.negation_detector.two_step import NegationDetector

detector = NegationDetector('output/negation_detector')
print('\nSample inference:')
examples = [
    "The treatment did not improve patient outcomes.",
    "We found no significant association between the mutation and disease progression.",
    "The drug was not effective, and there was no improvement in symptoms.",
]

for i, text in enumerate(examples, 1):
    print(f'\nExample {i}: {text}')
    result = detector.predict(text)
    if result['negations']:
        for j, neg in enumerate(result['negations'], 1):
            cue = neg['cue']
            scope = neg.get('scope')
            print(f"  Negation {j}:")
            print(f"    Cue: '{cue['text']}' at {cue['start']}-{cue['end']}")
            if scope:
                print(f"    Scope: '{scope['text']}' at {scope['start']}-{scope['end']}")
            else:
                print(f"    Scope: None")
    else:
        print("  No negations detected")
PY

python - <<'PY'
from huggingface_hub import upload_folder

repo_id = "thomaslim6793/indra_bert_negation_detector"
folder_path = "output/negation_detector"
ignore_patterns = ["cue_checkpoint", "scope_checkpoint", "cached_dataset", "cached_dataset/*"]

print("\nUploading trained model to Hugging Face Hub...")
upload_folder(
    repo_id=repo_id,
    folder_path=folder_path,
    commit_message="Update two-step negation detector",
    ignore_patterns=ignore_patterns,
)
print("Upload complete.")
PY
