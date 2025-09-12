module load cuda/12.3.0
module load anaconda3/2024.06

# Set up environment
cd /home/hy.lim/indra_bert
export CUDA_VISIBLE_DEVICES=0

# Activate conda
conda activate indra

python -m indra_bert.indra_stmt_classifier.train \
    --dataset_path data/indra_benchmark_annotated_data/indra_benchmark_corpus_annotated_stratified_sample.jsonl \
    --output_dir output/indra_stmt_classifier \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --epochs 8 \
    --version 1.1 \
    --max_negatives_per_positive 5
