import json
import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset

INPUT_JSONL = "data/indra_benchmark_corpus_annotated.jsonl"
OUTPUT_DIR = "data/re_dataset"  # Saved datasets will go here

# Label mapping
relationship_types = [
    'Phosphorylation', 'Autophosphorylation','Transphosphorylation', 
    'Dephosphorylation', 'Hydroxylation','Dehydroxylation', 
    'Sumoylation', 'Desumoylation', 'Acetylation',
    'Deacetylation', 'Glycosylation', 'Deglycosylation', 
    'Ribosylation', 'Deribosylation', 'Ubiquitination', 
    'Deubiquitination', 'Farnesylation', 'Defarnesylation', 
    'Geranylgeranylation', 'Degeranylgeranylation', 'Palmitoylation', 
    'Depalmitoylation', 'Myristoylation', 'Demyristoylation',
    'Methylation', 'Demethylation', 
    'Activation', 'Inhibition',
    'ActiveForm', 
    'Gef', 'Gap',
    'Complex', 
    'Translocation', 
    'IncreaseAmount', 'DecreaseAmount',
    'Conversion', 'Unresolved'
]
relation2id = {rel: i for i, rel in enumerate(relationship_types)}

# ---- Preprocess raw data ----
def load_and_preprocess_raw_data(input_path, tokenizer):
    examples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc="Loading and preprocessing")):
            obj = json.loads(line)
            text = obj["Updated Text"]
            label = obj["type"]

            encoding = tokenizer(text, add_special_tokens=False)
            tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

            examples.append({
                "id": idx,
                "tokens": tokens,
                "relation_label": label,
                "relation_label_id": relation2id[label]
            })

    return examples
# ---- Preprocessing for model input ----
def preprocess_examples_for_model(examples, tokenizer):
    encoding = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        add_special_tokens=True,
        padding="longest"
    )
    encoding["labels"] = examples["relation_label_id"]
    encoding["relation_label"] = examples["relation_label"]  # Optional, for debugging
    return encoding
