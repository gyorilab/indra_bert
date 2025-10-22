# INDRA BERT: Statement Extraction using Fine-tuned BERT Models

A comprehensive system for extracting structured biological statements from text using fine-tuned BERT models, designed to integrate with the [INDRA](https://github.com/sorgerlab/indra) knowledge assembly platform.

## Overview

INDRA BERT is a three-stage pipeline that extracts structured biological statements from scientific text:

1. **Named Entity Recognition (NER)**: Identifies biological entities (proteins, genes, etc.)
2. **Statement Classification**: Determines the type of relationship between entity pairs
3. **Role Assignment**: Assigns specific roles (subject, object, enzyme, etc.) to entities within statements

The system outputs structured statements in INDRA-compatible format, enabling integration with the broader INDRA ecosystem for knowledge assembly and reasoning.

## Features

- **Multi-stage Pipeline**: Combines NER, classification, and role assignment for comprehensive statement extraction
- **Batch Processing**: Efficient processing of multiple texts with optimized batching
- **INDRA Integration**: Native compatibility with INDRA statement format and processing pipeline
- **Pre-trained Models**: Ready-to-use models available on Hugging Face Hub
- **Configurable Confidence**: Adjustable confidence thresholds for quality control
- **Multiple Tokenization**: Support for NLTK and spaCy sentence tokenization

## Architecture

### Core Components

- **`IndraStructuredExtractor`**: Main pipeline class that orchestrates the three-stage extraction process
- **`AgentNERModel`**: Named entity recognition for biological entities
- **`IndraStmtClassifier`**: Statement type classification using custom BERT head
- **`IndraAgentsTagger`**: Role assignment for entities within statements

### Model Pipeline

```
Text Input → Sentence Tokenization → NER → Entity Pairing → Statement Classification → Role Assignment → Structured Output
```

## Installation

```bash
pip install indra-bert
```

### Dependencies

- Python >= 3.9
- PyTorch
- Transformers
- NLTK
- NumPy
- scikit-learn
- tqdm

## Quick Start

### Basic Usage

```python
from indra_bert import IndraStructuredExtractor

# Initialize the extractor with pre-trained models
extractor = IndraStructuredExtractor(
    ner_model_path="thomaslim6793/indra_bert_ner_agent_detection",
    stmt_model_path="thomaslim6793/indra_bert_indra_stmt_classifier",
    role_model_path="thomaslim6793/indra_bert_indra_stmt_agents_role_assigner",
    stmt_conf_threshold=0.95
)

# Extract statements from text
text = "AKT phosphorylates BAD at Ser136."
statements = extractor.extract_structured_statements(text)

# Get INDRA-compatible JSON format
indra_statements = extractor.get_json_indra_stmts(text)
```

### Batch Processing

```python
# Process multiple texts efficiently
texts = [
    "AKT phosphorylates BAD at Ser136.",
    "EGFR activates MAPK signaling pathway.",
    "p53 inhibits cell proliferation."
]

# Batch extraction
batch_statements = extractor.extract_structured_statements_batch(texts)
indra_batch = extractor.get_json_indra_stmts_batch(texts)
```

## Model Details

### Pre-trained Models

The system uses three fine-tuned BERT models:

1. **NER Model** (`thomaslim6793/indra_bert_ner_agent_detection`)
   - Token classification for biological entity detection
   - BIO tagging scheme for entity spans

2. **Statement Classifier** (`thomaslim6793/indra_bert_indra_stmt_classifier`)
   - Custom classification head for statement types
   - Supports various INDRA statement types (Phosphorylation, Activation, etc.)

3. **Role Assigner** (`thomaslim6793/indra_bert_indra_stmt_agents_role_assigner`)
   - Token classification for role assignment
   - Assigns specific roles to entities within statements

### Statement Types

The system recognizes various biological statement types including:
- Phosphorylation
- Dephosphorylation
- Activation
- Inhibition
- Binding
- And more...

## Output Format

### Structured Statements

```python
{
    'original_text': 'AKT phosphorylates BAD at Ser136.',
    'entity_pair': [
        {'text': 'AKT', 'start': 0, 'end': 3, 'label': 'PROTEIN'},
        {'text': 'BAD', 'start': 16, 'end': 19, 'label': 'PROTEIN'}
    ],
    'stmt_pred': {
        'label': 'Phosphorylation',
        'confidence': 0.98,
        'raw_output': {...}
    },
    'role_pred': {
        'roles': [
            {'role': 'enz', 'text': 'AKT', 'start': 0, 'end': 3},
            {'role': 'sub', 'text': 'BAD', 'start': 16, 'end': 19}
        ],
        'raw_output': {...}
    }
}
```

### INDRA JSON Format

```python
{
    "type": "Phosphorylation",
    "enz": {
        "name": "AKT",
        "db_refs": {"TEXT": "AKT"}
    },
    "sub": {
        "name": "BAD", 
        "db_refs": {"TEXT": "BAD"}
    },
    "evidence": [{
        "source_api": "indra_bert",
        "text": "AKT phosphorylates BAD at Ser136.",
        "annotations": {
            "agents": {
                "raw_text": ["AKT", "BAD"],
                "coords": [[0, 3], [16, 19]]
            }
        }
    }]
}
```

## Training

### Training Individual Models

Each component can be trained independently:

```bash
# Train NER model
cd indra_bert/ner_agent_detector
python train.py --train_data path/to/train.json --val_data path/to/val.json

# Train statement classifier
cd indra_bert/indra_stmt_classifier
python train.py --train_data path/to/train.json --val_data path/to/val.json

# Train role assigner
cd indra_bert/indra_agent_role_assigner
python train.py --train_data path/to/train.json --val_data path/to/val.json
```

### Data Format

Training data should be in JSON format with appropriate annotations for each model type. See the individual training scripts for detailed format specifications.

## Evaluation

The project includes benchmark notebooks and evaluation scripts:

- `benchmark.ipynb`: Performance evaluation on test datasets
- `experiment.ipynb`: Experimental analysis and comparison
- `stmt_classifier_basic_benchmark.py`: Basic benchmarking script

## Integration with INDRA

INDRA BERT is designed to integrate seamlessly with the INDRA platform:

```python
from indra.sources.indra_bert.api import process_texts

# Process texts and get INDRA statements
texts = ["AKT phosphorylates BAD at Ser136."]
results = process_texts(texts, stmt_conf_threshold=0.95)

# Access INDRA statement objects
for result in results:
    for stmt in result.statements:
        print(f"Statement type: {stmt.__class__.__name__}")
        print(f"Agents: {stmt.agent_list()}")
```

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the BSD-2 License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use INDRA BERT in your research, please cite:

```bibtex
@software{indra_bert,
  title={INDRA BERT: Statement Extraction using Fine-tuned BERT Models},
  author={Lim, Thomas and Gyori, Benjamin M.},
  year={2025},
  url={https://github.com/gyorilab/indra_bert}
}
```

## Acknowledgments

This work builds upon the [INDRA](https://github.com/sorgerlab/indra) platform and leverages state-of-the-art transformer models for biological text mining.
