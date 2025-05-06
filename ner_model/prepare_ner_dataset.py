import json
import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset

INPUT_JSONL = "data/indra_benchmark_corpus_annotated.jsonl"
OUTPUT_DIR = "data/ner_dataset"  # Saved datasets will go here

# ---- Label mapping ----
label2id = {"O": 0, "B-AGENT": 1, "I-AGENT": 2}
id2label = {v: k for k, v in label2id.items()}

def parse_agent_spans(text):
    """
    Parse the text and find agent spans.
    Returns:
    - clean_text: text with <Agent> and </Agent> removed
    - spans: list of (start_char, end_char) for agent spans
    """
    spans = []
    clean_text = ""

    pattern = re.compile(r"<Agent>(.*?)</Agent>")
    last_end = 0

    for match in pattern.finditer(text):
        start, end = match.span()
        agent_text = match.group(1)

        # Add text before this agent
        clean_text += text[last_end:start]
        agent_start = len(clean_text)
        clean_text += agent_text
        agent_end = len(clean_text)

        spans.append((agent_start, agent_end))
        last_end = end

    # Add the remaining text
    clean_text += text[last_end:]

    return clean_text, spans

def char_to_token_labels(tokens, token_offsets, agent_spans):
    """
    Given token offsets and agent spans, produce BIO labels.
    """
    labels = ["O"] * len(tokens)

    for span_start, span_end in agent_spans:
        for i, (tok_start, tok_end) in enumerate(token_offsets):
            if tok_end <= span_start:
                continue
            if tok_start >= span_end:
                continue
            # Token and span overlap
            if tok_start >= span_start and tok_end <= span_end:
                # Inside span
                labels[i] = "I-AGENT" if labels[i] != "B-AGENT" else labels[i]
                # If first token in span
                if tok_start == span_start:
                    labels[i] = "B-AGENT"

    return labels

def generate_examples(input_path, tokenizer):
    """
    Generator that yields tokenized examples with BIO tags.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc="Processing examples")):
            obj = json.loads(line)
            text = obj["Updated Text"]

            # Step 1: Parse agent spans and clean text
            clean_text, agent_spans = parse_agent_spans(text)

            # Step 2: Tokenize
            encoding = tokenizer(clean_text, return_offsets_mapping=True, add_special_tokens=False)
            tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
            offsets = encoding["offset_mapping"]

            # Step 3: Align spans to tokens -> BIO labels
            ner_tags = char_to_token_labels(tokens, offsets, agent_spans)

            # Step 4: Yield example
            yield {
                "id": idx,
                "tokens": tokens,
                "ner_tags": ner_tags,
                "ner_tags_id": [label2id[tag] for tag in ner_tags]
            }

# Create a generator function for the dataset for `Dataset.from_generator`
def generator_function(input_path, tokenizer):
    return lambda: generate_examples(input_path, tokenizer)

# Batched examples processing for alignment. Assumes that the 
# example from `generator_function` is made into a batch of examples.
# i.e. 'tokens' and 'ner_tags' are lists of lists instead of single lists.
def preprocess_examples(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], 
                                 is_split_into_words=True, 
                                 truncation=True, 
                                 padding="longest")
    all_labels = []

    for i in range(len(tokenized_inputs["input_ids"])):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        labels = []
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            else:
                labels.append(examples["ner_tags_id"][i][word_idx])
        all_labels.append(labels)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

# Main function to load dataset and save it
def main():
    from transformers import AutoTokenizer

    TOKENIZER_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

    # Load dataset from generator
    dataset = Dataset.from_generator(generator_function(INPUT_JSONL, tokenizer))

    # Show few examples
    print(dataset)

    # Save to disk
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(OUTPUT_DIR)
    print(f"Saved dataset to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
