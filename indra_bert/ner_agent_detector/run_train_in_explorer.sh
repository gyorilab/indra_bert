#!/bin/bash
#SBATCH --job-name=indra_bert_ner_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=indra_bert_ner_train_%j.out
#SBATCH --error=indra_bert_ner_train_%j.err

# Load modules
module load cuda/12.3.0
module load anaconda3/2024.06

# Set up environment
cd /home/hy.lim/indra_bert
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate indra

# Remove cached dataset so label-set changes take effect
rm -rf output/ner_agent_detection/cached_dataset

# Train the NER model with BioRED entity types
python -m indra_bert.ner_agent_detector.train \
    --train_data data/train/ner_data/BioRED/processed/train_biored_sent_annotated.json \
    --val_data data/train/ner_data/BioRED/processed/val_biored_sent_annotated.json \
    --test_data data/train/ner_data/BioRED/processed/test_biored_sent_annotated.json \
    --output_dir output/ner_agent_detection \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --epochs 8 \
    --version 1.1 \
    --use_cached_dataset

# Quick smoke test to confirm entity types are emitted
python - <<'PY'
from indra_bert.ner_agent_detector.inference import AgentNERExtractor

detector = AgentNERExtractor("output/ner_agent_detection")
text = "The interaction between BRCA1 and RAD51 is crucial for DNA repair."
result = detector.predict(text)

print("\nSample inference:")
print(result["annotated_text"])
print("Entities:")
for ent in result["entities"]:
    print(f" - {ent['text']} [{ent.get('type')} / {ent.get('raw_type')}] -> ({ent['start']}, {ent['end']})")
PY

echo "NER training and verification complete."
