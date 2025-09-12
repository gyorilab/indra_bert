__all__ = ['IndraStructuredExtractor']

import os
from pathlib import Path
from typing import List
from itertools import combinations

from huggingface_hub import hf_hub_download

from .ner_agent_detector.inference import AgentNERExtractor
from .indra_stmt_classifier.inference import IndraStmtClassifier
from .indra_agent_role_assigner.inference import IndraAgentsTagger
from .agent_mutation_detector.inference import AgentMutationDetector
from .utils.annotate import annotate_entities
import logging
logger = logging.getLogger(__name__)


class IndraStructuredExtractor:
    def __init__(self, 
                 ner_model_path="thomaslim6793/indra_bert_ner_agent_detection",
                 stmt_model_path="thomaslim6793/indra_bert_indra_stmt_classifier", 
                 role_model_path="thomaslim6793/indra_bert_indra_stmt_agents_role_assigner",
                 mutations_model_path="thomaslim6793/indra_bert_agent_mutation_detector",
                 stmt_conf_threshold=0.95):
        self.ner_model = AgentNERExtractor(ner_model_path)
        self.stmt_model = IndraStmtClassifier(stmt_model_path)
        self.role_model = IndraAgentsTagger(role_model_path)
        self.mutations_model = AgentMutationDetector(mutations_model_path)
        self.stmt_conf_threshold = stmt_conf_threshold

        self.ner_model_local_path = self._resolve_model_path(ner_model_path, "ner")
        self.stmt_model_local_path = self._resolve_model_path(stmt_model_path, "stmt")
        self.role_model_local_path = self._resolve_model_path(role_model_path, "role")
        self.mutations_model_local_path = self._resolve_model_path(mutations_model_path, "mutations")

    def _resolve_model_path(self, model_path, label="model"):
        try:
            path = hf_hub_download(repo_id=model_path, filename="config.json")
            return os.path.dirname(path)
        except Exception as e:
            logger.info(f"Not a huggingface hub repo id ({label}): {model_path}. "
                        f"Assuming local path...")
            local_path = Path(model_path)
            return local_path if local_path.is_absolute() else Path.cwd() / local_path

    def get_entity_pairs(self, entity_preds):
        return list(combinations(entity_preds['entity_spans'], 2))

    def extract_structured_statements(self, text):
        stmts = []
        sentences = self.sentence_tokenize(text, mode='nltk')
        for sentence in sentences:
            entity_preds = self.ner_model.predict(sentence)
            pairs = self.get_entity_pairs(entity_preds)

            for pair in pairs:
                annotated_text = annotate_entities(sentence, pair)
                stmt_pred = self.stmt_model.predict(annotated_text)
                stmt_conf = stmt_pred.get('confidence', 0.0)

                if stmt_conf < self.stmt_conf_threshold:
                    continue
                if stmt_pred['predicted_label'] == "No_Relation":
                    continue

                role_pred = self.role_model.predict(stmt_pred['predicted_label'], annotated_text)
                
                # Detect mutations for each agent in the pair
                mutations_pred = self.mutations_model.predict(pair, annotated_text)

                stmt = {
                    'original_text': sentence,
                    'entity_pair': pair,
                    'annotated_text': annotated_text,
                    'ner_info': {
                        'all_entities': entity_preds['entity_spans'],
                        'entity_pair': pair
                    },
                    'stmt_pred': {
                        'label': stmt_pred['predicted_label'],
                        'confidence': stmt_conf,
                        'raw_output': stmt_pred
                    },
                    'role_pred': {
                        'roles': role_pred.get('role_spans', []),
                        'raw_output': role_pred
                    },
                    'mutations_pred': {
                        'mutations': mutations_pred.get('mutations', {}),
                        'raw_output': mutations_pred
                    }
                }

                stmts.append(stmt)

        return stmts


    def extract_structured_statements_batch(self, text):
        """Efficiently process multiple texts using batching at each pipeline step."""
        all_statements = []

        # STEP 1: Run NER in batch
        sentences = self.sentence_tokenize(text, mode='nltk')
        ner_preds_batch = self.ner_model.predict_batch(sentences)

        # For each sentence, get entity pairs and prepare annotated texts for classification
        stmt_inputs = []
        stmt_pair_info = []
        for sentence, ner_preds in zip(sentences, ner_preds_batch):
            pairs = self.get_entity_pairs(ner_preds)
            for pair in pairs:
                annotated_text = annotate_entities(sentence, pair)
                stmt_inputs.append(annotated_text)
                stmt_pair_info.append((sentence, pair, ner_preds))  # to keep track later

        # STEP 2: Run statement classification in batch
        if not stmt_inputs:
            return []
        stmt_preds_batch = self.stmt_model.predict_batch(stmt_inputs)

        # Filter by confidence and collect inputs for role prediction
        role_inputs_text = []
        role_inputs_type = []
        final_pairs = []

        for i, stmt_pred in enumerate(stmt_preds_batch):
            conf = stmt_pred.get('confidence', 0.0)
            if conf >= self.stmt_conf_threshold:
                role_inputs_text.append(stmt_inputs[i])
                role_inputs_type.append(stmt_pred['predicted_label'])
                final_pairs.append((stmt_pred, stmt_pair_info[i]))

        # STEP 3: Run role assignment in batch
        role_preds_batch = self.role_model.predict_batch(role_inputs_type, role_inputs_text)

        # STEP 4: Run mutation detection in batch
        mutations_inputs_pairs = []
        mutations_inputs_text = []
        for (text, pair, ner_preds) in [x[1] for x in final_pairs]:
            mutations_inputs_pairs.append(pair)
            mutations_inputs_text.append(annotate_entities(text, pair))
        
        mutations_preds_batch = self.mutations_model.predict_batch(mutations_inputs_pairs, mutations_inputs_text)

        # STEP 5: Assemble final results
        for stmt_pred, (text, pair, ner_preds), role_pred, mutations_pred in zip(
                [x[0] for x in final_pairs],
                [x[1] for x in final_pairs],
                role_preds_batch,
                mutations_preds_batch):
            
            if stmt_pred['predicted_label'] == "No_Relation":
                continue

            stmt = {
                'original_text': text,
                'entity_pair': pair,
                'annotated_text': annotate_entities(text, pair),
                'ner_info': {
                    'all_entities': ner_preds['entity_spans'],
                    'entity_pair': pair
                },
                'stmt_pred': {
                    'label': stmt_pred['predicted_label'],
                    'confidence': stmt_pred['confidence'],
                    'raw_output': stmt_pred
                },
                'role_pred': {
                    'roles': role_pred.get('role_spans', []),
                    'raw_output': role_pred
                },
                'mutations_pred': {
                    'mutations': mutations_pred.get('mutations', {}),
                    'raw_output': mutations_pred
                }
            }

            all_statements.append(stmt)

        return all_statements
    
    def get_json_indra_stmts(self, text, source_api="indra_bert"):
        """Extract statements and convert to INDRA-style JSON with agent coords."""
        try:
            structured_statements = self.extract_structured_statements_batch(text)
        except Exception as e:
            logger.warning(f"Batch extraction failed. Falling back to iterative extraction. Error: {e}")
            structured_statements = self.extract_structured_statements(text)
        
        indra_statements = []

        for stmt in structured_statements:
            stmt_type = stmt['stmt_pred']['label']

            if stmt_type== "No_Relation":
                continue

            roles = stmt['role_pred']['roles']
            mutations_pred = stmt['mutations_pred']['mutations']

            agent_roles = {}
            raw_texts = []
            coords = []

            for role_info in roles:
                role = role_info['role']
                name = role_info['text']
                start = role_info['start']
                end = role_info['end']
                raw_texts.append(name)
                coords.append([start, end])

                agent_roles[role] = {
                    "name": name,
                    "db_refs": {
                        "TEXT": name
                    }
                }
                
                if mutations_pred.get((start, end, name), None):
                    agent_roles[role]["mutations"] = mutations_pred[(start, end, name)]

            indra_stmt = {
                "type": stmt_type,
                **agent_roles,
                "evidence": [{
                    "source_api": source_api,
                    "text": stmt['original_text'],
                    "annotations": {
                        "agents": {
                            "raw_text": raw_texts,
                            "coords": coords
                        }
                    }
                }]
            }

            indra_statements.append(indra_stmt)

        return indra_statements
    
    def sentence_tokenize(self, text, mode='nltk'):
        """Tokenize text into sentences using the specified mode."""
        if mode == 'nltk':
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)
        elif mode == 'spacy':
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            return [sent.text for sent in doc.sents]
        else:
            raise ValueError("Unsupported tokenization mode. Use 'nltk' or 'spacy'.")
