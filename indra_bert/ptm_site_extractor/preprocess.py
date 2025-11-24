import json
import re
import random
from pathlib import Path
from tqdm import tqdm
from transformers import PreTrainedTokenizer
import torch
import pandas as pd

# ---- Parse annotated text to extract agent and PTM site spans ----
def parse_annotated_text(text):
    """
    Parse text with <role>agent</role> and <role.site>site</role.site> tags.
    
    Args:
        text: Text like "The <enz>p300</enz> phosphorylates <sub>p27</sub> on <sub.site>Ser216</sub.site>..."
        
    Returns:
        clean_text: Text without any tags
        agent_spans: List of agent spans with role info
        site_spans: List of PTM site spans with associated_agent info
    """
    agent_spans = []
    site_spans = []
    clean_text = ""
    last_end = 0
    
    # Find all tagged spans (both agents and sites)
    for match in re.finditer(r"<([^<>]+?)>(.*?)</\1>", text):
        tag = match.group(1)
        span_text = match.group(2)
        start, end = match.span()
        
        # Add text before this span
        clean_text += text[last_end:start]
        span_start = len(clean_text)
        clean_text += span_text
        span_end = len(clean_text)
        
        # Determine if this is a PTM site or agent
        if tag.endswith('.site'):
            # This is a PTM site
            associated_agent = tag.split('.')[0]
            site_spans.append({
                "start": span_start,
                "end": span_end,
                "text": span_text,
                "associated_agent": associated_agent
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
    
    return clean_text, agent_spans, site_spans

# ---- Create training examples for each agent ----
def create_agent_training_examples(clean_text, agent_spans, site_spans, max_negative_examples_per_agent=0):
    """
    Create positive training examples for each agent.
    
    Positive examples: BIO tags for PTM sites of the target agent
    
    Args:
        clean_text: Text without any tags
        agent_spans: List of agent spans
        site_spans: List of PTM site spans
        max_negative_examples_per_agent: Not used (kept for compatibility, should be 0)
        
    Returns:
        List of training examples (positive examples only)
    """
    examples = []
    
    for target_agent in agent_spans:
        # Create text with only this agent tagged
        agent_text = clean_text[:target_agent["start"]] + f"<e>{target_agent['text']}</e>" + clean_text[target_agent["end"]:]
        
        # === POSITIVE EXAMPLE: Target agent's PTM sites ===
        positive_sites = []
        for site in site_spans:
            # Check if site is associated with this agent by role
            if site.get("associated_agent") == target_agent["role"]:
                # Adjust site positions based on where the agent tag was added
                if site["start"] < target_agent["start"]:
                    # Site is before the agent, no position change needed
                    adjusted_site = site.copy()
                elif site["start"] >= target_agent["end"]:
                    # Site is after the agent, adjust by the tag length difference
                    tag_length_diff = len(f"<e>{target_agent['text']}</e>") - len(target_agent['text'])
                    adjusted_site = {
                        "start": site["start"] + tag_length_diff,
                        "end": site["end"] + tag_length_diff,
                        "text": site["text"],
                        "associated_agent": site["associated_agent"]
                    }
                else:
                    # Site overlaps with agent, skip it
                    continue
                positive_sites.append(adjusted_site)
        
        positive_example = {
            "text": agent_text,
            "clean_text": clean_text,
            "agent": target_agent,
            "sites": positive_sites,
            "example_type": "positive"
        }
        examples.append(positive_example)
    
    return examples

# ---- Assign BIO tags for PTM site detection ----
def char_to_token_labels(tokens, token_offsets, site_spans, example_type="positive"):
    """
    Assign BIO tags to tokens for PTM site detection.
    
    Args:
        tokens: List of tokens
        token_offsets: List of (start, end) character positions for each token
        site_spans: List of PTM site spans
        example_type: "positive" or "negative" - determines how sites are labeled
        
    Returns:
        List of BIO labels for each token
    """
    labels = ["O"] * len(tokens)
    
    # For negative examples, sites should be labeled as "O" (not sites for target agent)
    # For positive examples, sites should be labeled as "B-site" or "I-site"
    if example_type == "negative":
        # In negative examples, all sites are labeled as "O" (not relevant to target agent)
        return labels
    
    # For positive examples, label sites as B-site/I-site
    for site in site_spans:
        start_char = site["start"]
        end_char = site["end"]
        
        # Find tokens that overlap with this site
        for i, (tok_start, tok_end) in enumerate(token_offsets):
            if tok_start is None or tok_end is None:
                continue
            if tok_end <= start_char or tok_start >= end_char:
                continue
            
            if tok_start == start_char:
                labels[i] = "B-site"
            else:
                labels[i] = "I-site"
    
    return labels

# ---- Load and preprocess training data from TSV ----
def load_tsv_training_data(input_path, max_negative_examples_per_agent=0):
    """
    Load TSV file with PTM site annotations and create training examples.
    
    Args:
        input_path: Path to TSV file with columns: sentence_id, sentence_text, protein_agent, 
                   modification_site, protein_start, protein_end, site_start, site_end
        max_negative_examples_per_agent: Maximum number of negative examples per agent
        
    Returns:
        List of training examples
    """
    examples = []
    
    print(f"Loading TSV format from {input_path}...")
    # Use on_bad_lines='skip' to skip malformed lines (some rows have tabs in sentence_text)
    df = pd.read_csv(input_path, sep="\t", dtype=str, keep_default_na=False, quoting=1, on_bad_lines='skip')
    
    # Group by sentence to collect all agents and sites
    sentence_groups = {}
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading training data"):
        sentence_id = row.get("sentence_id", "").strip('"')
        sentence_text = row.get("sentence_text", "").strip('"')
        protein_agent = row.get("protein_agent", "").strip('"')
        modification_site = row.get("modification_site", "").strip('"')
        protein_start = row.get("protein_start", "").strip('"')
        protein_end = row.get("protein_end", "").strip('"')
        site_start = row.get("site_start", "").strip('"')
        site_end = row.get("site_end", "").strip('"')
        
        if not sentence_text or not protein_agent or not modification_site:
            continue
        
        # Convert string positions to int
        try:
            protein_start = int(protein_start) if protein_start else None
            protein_end = int(protein_end) if protein_end else None
            site_start = int(site_start) if site_start else None
            site_end = int(site_end) if site_end else None
        except (ValueError, TypeError):
            continue
        
        if sentence_id not in sentence_groups:
            sentence_groups[sentence_id] = {
                "text": sentence_text,
                "agents": [],
                "sites": []
            }
        
        # Add agent (assuming object/substrate role for PTM)
        # Use tuple for deduplication check
        agent_key = (protein_start, protein_end, protein_agent)
        agent_entry = {
            "start": protein_start,
            "end": protein_end,
            "text": protein_agent,
            "role": "object"  # PTM sites are typically on the substrate/object
        }
        
        # Check if agent already exists (by position and text)
        existing_agent_keys = {(a["start"], a["end"], a["text"]) for a in sentence_groups[sentence_id]["agents"]}
        if agent_key not in existing_agent_keys:
            sentence_groups[sentence_id]["agents"].append(agent_entry)
        
        # Add site and associate it with the specific agent
        # Store as role string (like mutation detector) for Arrow compatibility
        site_entry = {
            "start": site_start,
            "end": site_end,
            "text": modification_site,
            "associated_agent": "object"  # Store as role string for Arrow compatibility
        }
        
        # Check for duplicate sites (same position and text)
        existing_site_keys = {(s["start"], s["end"], s["text"]) for s in sentence_groups[sentence_id]["sites"]}
        site_key = (site_start, site_end, modification_site)
        if site_key not in existing_site_keys:
            sentence_groups[sentence_id]["sites"].append(site_entry)
    
    # Create training examples for each sentence
    for sentence_id, group in tqdm(sentence_groups.items(), desc="Creating training examples"):
        clean_text = group["text"]
        agent_spans = group["agents"]
        site_spans = group["sites"]
        
        # Create training examples for each agent
        agent_examples = create_agent_training_examples(clean_text, agent_spans, site_spans, max_negative_examples_per_agent)
        examples.extend(agent_examples)
    
    return examples

# ---- Load and preprocess training data from JSON/JSONL ----
def load_and_preprocess_training_data(input_path, pubtator3_format=False, max_negative_examples_per_agent=0):
    """
    Load JSONL or JSON file and preprocess for PTM site detection training.
    
    Args:
        input_path: Path to JSONL file (INDRA format) or JSON file (PubTator3 format)
        pubtator3_format: If True, expect JSON array format from PubTator3.
                         If False, expect JSONL format from INDRA.
        max_negative_examples_per_agent: Maximum number of negative examples per agent
        
    Returns:
        List of training examples
    """
    examples = []
    
    if pubtator3_format:
        # Load JSON array format (PubTator3)
        print(f"Loading PubTator3 format from {input_path}...")
        with open(input_path, 'r') as f:
            data_list = json.load(f)
        
        for data in tqdm(data_list, desc="Loading training data"):
            # Parse the annotated text
            clean_text, agent_spans, site_spans = parse_annotated_text(data["annotated_text"])
            
            # Create training examples for each agent
            agent_examples = create_agent_training_examples(clean_text, agent_spans, site_spans, max_negative_examples_per_agent)
            examples.extend(agent_examples)
    else:
        # Load JSONL format (INDRA)
        print(f"Loading INDRA JSONL format from {input_path}...")
        with open(input_path, 'r') as f:
            for line in tqdm(f, desc="Loading training data"):
                if line.strip():
                    data = json.loads(line)
                    
                    # Parse the annotated text
                    clean_text, agent_spans, site_spans = parse_annotated_text(data["annotated_text"])
                    
                    # Create training examples for each agent
                    agent_examples = create_agent_training_examples(clean_text, agent_spans, site_spans, max_negative_examples_per_agent)
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
        label_set.add("B-site")
        label_set.add("I-site")
    
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
        sites = example["sites"]
        example_type = example.get("example_type", "positive")
        
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
        
        # Assign BIO labels based on example type
        labels = char_to_token_labels(tokens, token_offsets, sites, example_type)
        
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
            "sites": sites,
            "example_type": example_type,
            "agent": example.get("agent"),
            "offset_mapping": token_offsets
        }
        
        tokenized_examples.append(tokenized_example)
    
    return tokenized_examples

# ---- Main preprocessing function ----
def preprocess_for_training(input_path, tokenizer, max_length=512, tsv_format=True, max_negative_examples_per_agent=0, max_total_examples=None):
    """
    Main function to preprocess data for PTM site detection training.
    
    Args:
        input_path: Path to TSV file, JSONL file (INDRA), or JSON file (PubTator3) with PTM site annotations
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        tsv_format: If True, expect TSV format
        pubtator3_format: If True, expect JSON array format from PubTator3
        max_negative_examples_per_agent: Maximum negative examples per agent
        max_total_examples: Maximum total examples to use (None = use all)
        
    Returns:
        tokenized_examples: List of tokenized training examples
        label2id: Label to ID mapping
        id2label: ID to label mapping
    """
    # Load and preprocess data (TSV format only)
    examples = load_tsv_training_data(input_path, max_negative_examples_per_agent=max_negative_examples_per_agent)
    
    # Sample examples BEFORE tokenization if max_total_examples is specified
    if max_total_examples is not None and len(examples) > max_total_examples:
        print(f"Sampling {max_total_examples} examples from {len(examples)} total examples")
        import random
        random.shuffle(examples)
        examples = examples[:max_total_examples]
    
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
