import json
import re
from pathlib import Path
from tqdm import tqdm
from transformers import PreTrainedTokenizer
import torch

# ---- Parse annotated text to extract agent and mutation spans ----
def parse_annotated_text(text):
    """
    Parse text with <role>agent</role> and <role.variant>mutation</role.variant> tags.
    
    Args:
        text: Text like "The <enz>p300</enz> <enz.variant>E1242K</enz.variant> protein..."
        
    Returns:
        clean_text: Text without any tags
        agent_spans: List of agent spans with role info
        mutation_spans: List of mutation spans with role info
    """
    agent_spans = []
    mutation_spans = []
    clean_text = ""
    last_end = 0
    
    # Find all tagged spans (both agents and mutations)
    for match in re.finditer(r"<([^<>]+?)>(.*?)</\1>", text):
        tag = match.group(1)
        span_text = match.group(2)
        start, end = match.span()
        
        # Add text before this span
        clean_text += text[last_end:start]
        span_start = len(clean_text)
        clean_text += span_text
        span_end = len(clean_text)
        
        # Determine if this is a mutation or agent
        if tag.endswith('.variant'):
            # This is a mutation
            role = tag.split('.')[0]
            mutation_spans.append({
                "start": span_start,
                "end": span_end,
                "text": span_text,
                "role": role
            })
        else:
            # This is an agent
            agent_spans.append({
                "start": span_start,
                "end": span_end,
                "text": span_text,
                "role": tag
            })
        
        last_end = end
    
    # Add remaining text
    clean_text += text[last_end:]
    
    return clean_text, agent_spans, mutation_spans

# ---- Create training examples for each agent ----
def create_agent_training_examples(clean_text, agent_spans, mutation_spans):
    """
    Create training examples where each agent gets its own example.
    
    Args:
        clean_text: Text without any tags
        agent_spans: List of agent spans
        mutation_spans: List of mutation spans
        
    Returns:
        List of training examples, one per agent
    """
    examples = []
    
    for agent in agent_spans:
        # Create text with only this agent tagged
        agent_text = clean_text[:agent["start"]] + f"<e>{agent['text']}</e>" + clean_text[agent["end"]:]
        
        # Find mutations associated with this agent and adjust their positions
        agent_mutations = []
        for mutation in mutation_spans:
            if mutation["role"] == agent["role"]:
                # Adjust mutation positions based on where the agent tag was added
                if mutation["start"] < agent["start"]:
                    # Mutation is before the agent, no position change needed
                    adjusted_mutation = mutation.copy()
                elif mutation["start"] >= agent["end"]:
                    # Mutation is after the agent, adjust by the tag length difference
                    tag_length_diff = len(f"<e>{agent['text']}</e>") - len(agent['text'])
                    adjusted_mutation = {
                        "start": mutation["start"] + tag_length_diff,
                        "end": mutation["end"] + tag_length_diff,
                        "text": mutation["text"],
                        "role": mutation["role"]
                    }
                else:
                    # Mutation overlaps with agent, skip it
                    continue
                agent_mutations.append(adjusted_mutation)
        
        example = {
            "text": agent_text,
            "clean_text": clean_text,
            "agent": agent,
            "mutations": agent_mutations
        }
        examples.append(example)
    
    return examples

# ---- Assign BIO tags for mutation detection ----
def char_to_token_labels(tokens, token_offsets, mutation_spans):
    """
    Assign BIO tags to tokens for mutation detection.
    
    Args:
        tokens: List of tokens
        token_offsets: List of (start, end) character positions for each token
        mutation_spans: List of mutation spans
        
    Returns:
        List of BIO labels for each token
    """
    labels = ["O"] * len(tokens)
    
    for mutation in mutation_spans:
        start_char = mutation["start"]
        end_char = mutation["end"]
        
        # Find tokens that overlap with this mutation
        for i, (tok_start, tok_end) in enumerate(token_offsets):
            if tok_start is None or tok_end is None:
                continue
            if tok_end <= start_char or tok_start >= end_char:
                continue
            
            if tok_start == start_char:
                labels[i] = "B-mutation"
            else:
                labels[i] = "I-mutation"
    
    return labels

# ---- Load and preprocess training data ----
def load_and_preprocess_training_data(input_path):
    """
    Load JSONL file and preprocess for mutation detection training.
    
    Args:
        input_path: Path to JSONL file with variant annotations
        
    Returns:
        List of training examples
    """
    examples = []
    
    with open(input_path, 'r') as f:
        for line in tqdm(f, desc="Loading training data"):
            if line.strip():
                data = json.loads(line)
                
                # Parse the annotated text
                clean_text, agent_spans, mutation_spans = parse_annotated_text(data["annotated_text"])
                
                # Create training examples for each agent
                agent_examples = create_agent_training_examples(clean_text, agent_spans, mutation_spans)
                examples.extend(agent_examples)
    
    return examples

# ---- Build label mappings ----
def build_label_mappings(examples):
    """
    Build label2id and id2label mappings from training examples.
    
    Args:
        examples: List of training examples
        
    Returns:
        label2id: Dictionary mapping labels to IDs
        id2label: Dictionary mapping IDs to labels
    """
    label_set = set()
    
    for example in examples:
        # Add standard labels
        label_set.add("O")
        label_set.add("B-mutation")
        label_set.add("I-mutation")
    
    # Create mappings
    label2id = {label: i for i, label in enumerate(sorted(label_set))}
    id2label = {i: label for label, i in label2id.items()}
    
    return label2id, id2label

# ---- Tokenize and create dataset ----
def tokenize_examples(examples, tokenizer, label2id, max_length=512):
    """
    Tokenize examples and create training dataset.
    
    Args:
        examples: List of training examples
        tokenizer: HuggingFace tokenizer
        label2id: Label to ID mapping
        max_length: Maximum sequence length
        
    Returns:
        List of tokenized examples
    """
    tokenized_examples = []
    
    for example in tqdm(examples, desc="Tokenizing examples"):
        text = example["text"]
        mutations = example["mutations"]
        
        # Tokenize
        encoding = tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze(0))
        token_offsets = encoding["offset_mapping"].squeeze(0).tolist()
        
        # Assign BIO labels
        labels = char_to_token_labels(tokens, token_offsets, mutations)
        
        # Convert labels to IDs using the correct mapping
        label_ids = [label2id["O"]] * len(tokens)  # Default to "O"
        for i, label in enumerate(labels):
            label_ids[i] = label2id[label]
        
        tokenized_example = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_ids),
            "tokens": tokens,
            "text": text,
            "mutations": mutations
        }
        
        tokenized_examples.append(tokenized_example)
    
    return tokenized_examples

# ---- Main preprocessing function ----
def preprocess_for_training(input_path, tokenizer, max_length=512):
    """
    Main function to preprocess data for mutation detection training.
    
    Args:
        input_path: Path to JSONL file with variant annotations
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        
    Returns:
        tokenized_examples: List of tokenized training examples
        label2id: Label to ID mapping
        id2label: ID to label mapping
    """
    # Load and preprocess data
    examples = load_and_preprocess_training_data(input_path)
    
    # Build label mappings
    label2id, id2label = build_label_mappings(examples)
    
    # Tokenize examples
    tokenized_examples = tokenize_examples(examples, tokenizer, label2id, max_length)
    
    return tokenized_examples, label2id, id2label

# ---- Preprocessing for inference ----
def preprocess_for_inference(text, tokenizer, max_length=512):
    """
    Preprocess text for inference (no labels needed).
    
    Args:
        text: Input text
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with tokenized inputs
    """
    encoding = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    
    return encoding
