__all__ = ['IndraStructuredExtractor']

import os
from pathlib import Path
from typing import List, Optional, Union
from itertools import combinations

from huggingface_hub import hf_hub_download

from .ner_agent_detector.inference import AgentNERExtractor
from .indra_stmt_classifier.inference import IndraStmtClassifier
from .indra_agent_role_assigner.inference import IndraAgentsTagger
from .agent_mutation_detector.inference import AgentMutationDetector
from .ptm_site_extractor.inference import PTMSiteExtractor
from .utils.annotate import annotate_entities
from .utils.semantic_type_filter import TypeConstraintConfig, filter_statements_by_type_constraints
from .utils.parse_mutation import convert_to_indra_mutations
import logging
logger = logging.getLogger(__name__)

# PTM statement types that require PTM site extraction (from indra_schema.json Modification.pattern)
PTM_STMT_TYPES = {
    "Phosphorylation", "Dephosphorylation",
    "Autophosphorylation", "Transphosphorylation",
    "Ubiquitination", "Deubiquitination",
    "Sumoylation", "Desumoylation",
    "Hydroxylation", "Dehydroxylation",
    "Acetylation", "Deacetylation",
    "Glycosylation", "Deglycosylation",
    "Farnesylation", "Defarnesylation",
    "Geranylgeranylation", "Degeranylgeranylation",
    "Palmitoylation", "Depalmitoylation",
    "Myristoylation", "Demyristoylation",
    "Ribosylation", "Deribosylation",
    "Methylation", "Demethylation"
}

# Mapping for all compatible mapping from gate2 labels to gate3 (INDRA type) labels.
GATE2_TO_INDRA_TYPE = {
    # BioRED labels
    # Association: generic, non-directional relatedness.
    "Association": ["PTM"],
    # Bind: physical binding / complex formation between entities.
    "Bind": ["Complex"],
    # Comparison: comparative statement linking two entities (e.g., higher/lower).
    "Comparison": [],
    # Conversion: biochemical conversion / metabolic transformation.
    "Conversion": ["Conversion"],
    # Cotreatment: co-administration / combined treatment scenario.
    "Cotreatment": [],
    # Drug_Interaction: pharmacologic or biochemical interaction between drugs and proteins.
    "Drug_Interaction": [],
    # Negative_Correlation: entities covary inversely in text.
    "Negative_Correlation": ["PTM"],
    # Positive_Correlation: entities covary positively in text.
    "Positive_Correlation": ["PTM"],

    # BC5CDR labels
    # CID: chemical-induced disease relationship.
    "CID": [],

    # ChemProt labels
    # CPR:1 – PART_OF.
    "CPR:1": ["PTM", "Complex"],
    # CPR:2 – REGULATOR | DIRECT_REGULATOR | INDIRECT_REGULATOR.
    "CPR:2": ["PTM", "Complex", "Activation", "Inhibition", "IncreaseAmount", "DecreaseAmount"],
    # CPR:3 – UPREGULATOR | ACTIVATOR | INDIRECT_UPREGULATOR.
    "CPR:3": ["PTM", "Activation", "IncreaseAmount"],
    # CPR:4 – DOWNREGULATOR | INHIBITOR | INDIRECT_DOWNREGULATOR.
    "CPR:4": ["PTM", "Inhibition", "DecreaseAmount"],
    # CPR:5 – AGONIST | AGONIST-ACTIVATOR | AGONIST-INHIBITOR.
    "CPR:5": ["Complex", "Activation"],
    # CPR:6 – ANTAGONIST (blocking ligand).
    "CPR:6": ["Complex", "Inhibition"],
    # CPR:7 – MODULATOR | MODULATOR-ACTIVATOR | MODULATOR-INHIBITOR (context-dependent regulator).
    "CPR:7": ["PTM", "Complex", "Activation", "Inhibition", "IncreaseAmount", "DecreaseAmount"],
    # CPR:8 – COFACTOR (required binding partner).
    "CPR:8": ["Complex"],
    # CPR:9 – SUBSTRATE | PRODUCT_OF | SUBSTRATE_PRODUCT_OF (primary biotransformation relation).
    "CPR:9": ["PTM", "Conversion"],
    # CPR:10 – NOT (no meaningful relation).
"CPR:10": []
}

GATE1_HAS_RELATION_THRESHOLD = 0.45
GATE2_NO_RELATION_THRESHOLD = 0.9
GATE3_CONF_THRESHOLD = 0.8


def _has_required_roles(stmt_type: str, agent_roles: dict) -> bool:
    """Ensure required semantic roles are present before emitting a statement."""
    if not agent_roles:
        return False

    if stmt_type in PTM_STMT_TYPES:
        has_enz = "enz" in agent_roles
        has_sub = any(role in agent_roles for role in ("sub", "substrate", "object"))
        return has_enz and has_sub

    if stmt_type in {"Activation", "Inhibition", "IncreaseAmount", "DecreaseAmount"}:
        has_subj = "subj" in agent_roles
        has_obj = any(role in agent_roles for role in ("obj", "object"))
        return has_subj and has_obj

    if stmt_type in {"Complex", "Association"}:
        member_count = sum(1 for role in agent_roles if role.startswith("members"))
        return member_count >= 2

    if stmt_type == "Translocation":
        return "agent" in agent_roles

    return True


class IndraStructuredExtractor:
    def __init__(
        self,
        ner_model_path="thomaslim6793/indra_bert_ner_agent_detection",
        stmt_model_path="thomaslim6793/indra_bert_indra_stmt_classifier",
        role_model_path="thomaslim6793/indra_bert_indra_stmt_agents_role_assigner",
        mutations_model_path="thomaslim6793/indra_bert_agent_mutation_detection",
        ptm_site_model_path="thomaslim6793/indra_bert_ptm_site_extractor",
    ):
        self.ner_model = AgentNERExtractor(ner_model_path)
        self.stmt_model = IndraStmtClassifier(stmt_model_path)
        self.role_model = IndraAgentsTagger(role_model_path)
        self.mutations_model = AgentMutationDetector(mutations_model_path)
        self.ptm_site_model = PTMSiteExtractor(ptm_site_model_path) if ptm_site_model_path else None

        self.ner_model_local_path = self._resolve_model_path(ner_model_path, "ner")
        self.stmt_model_local_path = self._resolve_model_path(stmt_model_path, "stmt")
        self.role_model_local_path = self._resolve_model_path(role_model_path, "role")
        self.mutations_model_local_path = self._resolve_model_path(mutations_model_path, "mutations")
        if ptm_site_model_path:
            self.ptm_site_model_local_path = self._resolve_model_path(ptm_site_model_path, "ptm_site")

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
        entities = entity_preds.get('entities') or entity_preds.get('entity_spans') or []
        return list(combinations(entities, 2))

    def extract_structured_statements(self, text):
        stmts = []
        sentences = self.sentence_tokenize(text, mode='nltk')
        for sentence in sentences:
            entity_preds = self.ner_model.predict(sentence)
            pairs = self.get_entity_pairs(entity_preds)

            for pair in pairs:
                annotated_text = annotate_entities(sentence, pair)
                stmt_pred = self.stmt_model.predict(annotated_text)
                stmt_label = stmt_pred.get('gate3_prediction') or "unknown"

                role_pred = self.role_model.predict(stmt_label, annotated_text)
                
                # Detect mutations for each agent in the pair
                mutations_pred = self.mutations_model.predict(list(pair), annotated_text)
                
                # Detect PTM sites conditionally for PTM statement types
                ptm_sites_pred = None
                if self.ptm_site_model and stmt_label in PTM_STMT_TYPES:
                    # Get object/substrate agents from role prediction
                    object_agents = [r for r in role_pred.get('role_spans', []) 
                                   if r.get('role') in ('object', 'substrate', 'sub')]
                    if object_agents:
                        # Use predict_batch for consistency (even with single item)
                        ptm_sites_pred = self.ptm_site_model.predict_batch([object_agents], [annotated_text])[0]

                stmt = {
                    'original_text': sentence,
                    'entity_pair': pair,
                    'annotated_text': annotated_text,
                    'ner_info': {
                        'all_entities': entity_preds.get('entities') or entity_preds.get('entity_spans', []),
                        'entity_pair': pair
                    },
                    'stmt_label': stmt_label,
                    'stmt_pred': stmt_pred,
                    'role_pred': {
                        'roles': role_pred.get('role_spans', []),
                        'raw_output': role_pred
                    },
                    'mutations_pred': {
                        'mutations': mutations_pred.get('mutations', {}),
                        'raw_output': mutations_pred
                    }
                }
                
                if ptm_sites_pred is not None:
                    stmt['ptm_sites_pred'] = {
                        'sites': ptm_sites_pred.get('sites', []),
                        'raw_output': ptm_sites_pred
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
            stmt_label = stmt_pred.get('gate3_prediction') or "unknown"

            role_inputs_text.append(stmt_inputs[i])
            role_inputs_type.append(stmt_label)
            final_pairs.append({
                "stmt_label": stmt_label,
                "prediction": stmt_pred,
                "pair_info": stmt_pair_info[i],
                "annotated_text": stmt_inputs[i],
            })

        # STEP 3: Run role assignment in batch
        role_preds_batch = self.role_model.predict_batch(role_inputs_type, role_inputs_text)

        # STEP 4: Run mutation detection in batch
        mutations_inputs_pairs = []
        mutations_inputs_text = []
        for pair_data in final_pairs:
            sentence, pair, _ = pair_data["pair_info"]
            mutations_inputs_pairs.append(list(pair))
            mutations_inputs_text.append(pair_data["annotated_text"])
        
        mutations_preds_batch = self.mutations_model.predict_batch(mutations_inputs_pairs, mutations_inputs_text)

        # STEP 5: Run PTM site extraction conditionally for PTM statement types
        ptm_site_preds_batch = [None] * len(final_pairs)
        if self.ptm_site_model:
            ptm_stmt_indices = []
            ptm_inputs_agents = []
            ptm_inputs_text = []
            
            for i, pair_data in enumerate(final_pairs):
                stmt_type = pair_data["stmt_label"]
                if stmt_type in PTM_STMT_TYPES:
                    # For PTM statements, extract sites from object/substrate agent
                    # Get object agent from role_pred
                    role_pred = role_preds_batch[i]
                    object_agents = [r for r in role_pred.get('role_spans', []) 
                                   if r.get('role') in ('object', 'substrate', 'sub')]
                    
                    if object_agents:
                        ptm_stmt_indices.append(i)
                        ptm_inputs_agents.append(object_agents)
                        ptm_inputs_text.append(pair_data["annotated_text"])
            
            if ptm_inputs_agents:
                ptm_results = self.ptm_site_model.predict_batch(ptm_inputs_agents, ptm_inputs_text)
                for idx, result in zip(ptm_stmt_indices, ptm_results):
                    ptm_site_preds_batch[idx] = result

        # STEP 6: Assemble final results
        for i, (pair_data, role_pred, mutations_pred) in enumerate(zip(
                final_pairs,
                role_preds_batch,
                mutations_preds_batch)):
            stmt_label = pair_data["stmt_label"]
            stmt_pred = pair_data["prediction"]
            text, pair, ner_preds = pair_data["pair_info"]

            stmt = {
                'original_text': text,
                'entity_pair': pair,
                'annotated_text': pair_data["annotated_text"],
                'ner_info': {
                    'all_entities': ner_preds.get('entities') or ner_preds.get('entity_spans', []),
                    'entity_pair': pair
                },
                'stmt_label': stmt_label,
                'stmt_pred': stmt_pred,
                'role_pred': {
                    'roles': role_pred.get('role_spans', []),
                    'raw_output': role_pred
                },
                'mutations_pred': {
                    'mutations': mutations_pred.get('mutations', {}),
                    'raw_output': mutations_pred
                }
            }
            
            # Add PTM site predictions if available
            if ptm_site_preds_batch[i] is not None:
                stmt['ptm_sites_pred'] = {
                    'sites': ptm_site_preds_batch[i].get('sites', []),
                    'raw_output': ptm_site_preds_batch[i]
                }

            all_statements.append(stmt)

        return all_statements
    
    def get_json_indra_stmts(
        self,
        text,
        source_api="indra_bert",
        semantic_type_filter: bool = True,
        semantic_type_filter_config: Optional[Union[TypeConstraintConfig, dict]] = None,
    ):
        """Extract statements and convert to INDRA-style JSON with agent coords."""
        try:
            structured_statements = self.extract_structured_statements_batch(text)
        except Exception as e:
            logger.warning(f"Batch extraction failed. Falling back to iterative extraction. Error: {e}")
            structured_statements = self.extract_structured_statements(text)
 
        if semantic_type_filter:
            if semantic_type_filter_config is None:
                cfg = TypeConstraintConfig()
            elif isinstance(semantic_type_filter_config, TypeConstraintConfig):
                cfg = semantic_type_filter_config
            elif isinstance(semantic_type_filter_config, dict):
                cfg = TypeConstraintConfig(**semantic_type_filter_config)
            else:
                raise TypeError(
                    "semantic_type_filter_config must be a TypeConstraintConfig, dict, or None"
                )

            filter_result = filter_statements_by_type_constraints(structured_statements, config=cfg)
            if filter_result.drop_reasons:
                logger.debug(
                    "Semantic type filtering dropped statements: %s",
                    dict(filter_result.drop_reasons),
                )
            structured_statements = filter_result.kept_statements

        indra_statements = []

        for stmt in structured_statements:
            raw_pred = stmt.get('stmt_pred', {})
            gate1_prediction = raw_pred.get('gate1_prediction')
            gate2_prediction = raw_pred.get('gate2_prediction')
            gate3_prediction = raw_pred.get('gate3_prediction')
            gate1_probs = raw_pred.get('gate1_probs') or {}
            gate2_probs = raw_pred.get('gate2_probs') or {}
            gate3_probs = raw_pred.get('gate3_probs') or {}

            gate1_has_prob = float(gate1_probs.get("has_relation", 0.0))
            gate1_positive = gate1_prediction == "has_relation" or gate1_has_prob >= GATE1_HAS_RELATION_THRESHOLD
            if not gate1_positive:
                continue

            gate2_no_prob = float(gate2_probs.get("no_relation", 0.0))
            gate2_prediction_effective = gate2_prediction

            if gate2_prediction == "no_relation":
                if gate2_no_prob >= GATE2_NO_RELATION_THRESHOLD:
                    continue
                else:
                    non_no_candidates = [
                        (label, prob) for label, prob in gate2_probs.items() if label != "no_relation"
                    ]
                    if non_no_candidates:
                        gate2_prediction_effective = max(non_no_candidates, key=lambda x: x[1])[0]
                    else:
                        gate2_prediction_effective = None

            if gate2_prediction_effective is None:
                continue

            gate2_type_candidates = GATE2_TO_INDRA_TYPE.get(gate2_prediction_effective, [])
            stmt_type = None

            if "PTM" in gate2_type_candidates:
                if gate3_prediction in PTM_STMT_TYPES:
                    stmt_type = gate3_prediction
                else:
                    continue
            elif gate2_type_candidates:
                if gate3_prediction in gate2_type_candidates:
                    stmt_type = gate3_prediction
                else:
                    stmt_type = gate2_type_candidates[0]
            else:
                continue

            if stmt_type in (None, "no_relation","No_Relation", "unknown"):
                continue

            stmt_confidence = 0.0
            if isinstance(gate3_probs, dict):
                stmt_confidence = float(gate3_probs.get(stmt_type, gate3_probs.get(str(stmt_type), 0.0)))
            if stmt_confidence < GATE3_CONF_THRESHOLD:
                continue

            roles = stmt['role_pred']['roles']
            ner_info = stmt.get('ner_info', {})
            ner_entities = ner_info.get('all_entities') or []
            ner_span_index = {
                (ent.get('start'), ent.get('end')): ent
                for ent in ner_entities
                if ent.get('start') is not None and ent.get('end') is not None
            }

            reconciled_roles = []
            for role_info in roles:
                start = role_info.get('start')
                end = role_info.get('end')
                text = role_info.get('text')
                role = role_info.get('role')

                ner_ent = ner_span_index.get((start, end))
                if ner_ent is None:
                    ner_ent = next(
                        (e for e in ner_entities
                         if e.get('text') == text and e.get('start') is not None and e.get('end') is not None),
                        None
                    )
                if ner_ent is None:
                    logger.debug(
                        "Dropping role span with no matching NER entity: role=%s, span=(%s, %s, %s)",
                        role, start, end, text
                    )
                    continue

                reconciled_roles.append({
                    "role": role,
                    "start": ner_ent.get('start', start),
                    "end": ner_ent.get('end', end),
                    "text": ner_ent.get('text', text),
                    "type": ner_ent.get('type'),
                    "raw_type": ner_ent.get('raw_type')
                })

            if not reconciled_roles:
                continue

            mutations_pred = stmt['mutations_pred']['mutations']
            ptm_sites = stmt.get('ptm_sites_pred', {}).get('sites', [])

            agent_roles = {}
            raw_texts = []
            coords = []
            ptm_sites_list = []

            for role_info in reconciled_roles:
                role = role_info['role']
                name = role_info['text']
                start = role_info['start']
                end = role_info['end']

                raw_texts.append(name)
                coords.append([start, end])

                agent_roles[role] = {
                    "name": name,
                    "type": role_info.get('type'),
                    "raw_type": role_info.get('raw_type'),
                    "db_refs": {
                        "TEXT": name
                    }
                }
                
                mutation_key = (start, end, name)
                if mutation_key not in mutations_pred:
                    mutation_key = next(
                        (key for key in mutations_pred.keys()
                         if len(key) == 3 and key[0] == start and key[1] == end),
                        mutation_key
                    )

                if mutations_pred.get(mutation_key):
                    raw_mutations = mutations_pred[mutation_key]
                    parsed_mutations = convert_to_indra_mutations(raw_mutations)
                    agent_roles[role]["mutations"] = parsed_mutations

            # Collect PTM sites for object/substrate agents using reconciled spans
            for ptm_site in ptm_sites:
                site_agent = ptm_site.get('agent', {})
                agent_key = (site_agent.get('start'), site_agent.get('end'), site_agent.get('text'))
                for role_info in reconciled_roles:
                    if (role_info['start'], role_info['end'], role_info['text']) == agent_key:
                        if role_info.get('role') in ('object', 'substrate', 'sub'):
                            ptm_sites_list.append({
                                "residue": ptm_site.get('residue'),
                                "position": ptm_site.get('position'),
                                "normalized": ptm_site.get('normalized'),
                                "text": ptm_site.get('text'),
                                "agent_role": role_info['role']
                            })
                            if "mods" not in agent_roles[role_info['role']]:
                                agent_roles[role_info['role']]["mods"] = []
                            agent_roles[role_info['role']]["mods"].append({
                                "mod_type": stmt_type.lower(),
                                "residue": ptm_site.get('residue'),
                                "position": ptm_site.get('position')
                            })
                            break

            if not _has_required_roles(stmt_type, agent_roles):
                logger.debug(
                    "Skipping %s statement due to missing required roles: %s",
                    stmt_type,
                    list(agent_roles.keys()),
                )
                continue

            evidence_annotations = {
                "agents": {
                    "raw_text": raw_texts,
                    "coords": coords
                }
            }
            
            if ptm_sites_list:
                evidence_annotations["ptm_sites"] = ptm_sites_list

            indra_stmt = {
                "type": stmt_type,
                **agent_roles,
                "evidence": [{
                    "source_api": source_api,
                    "text": stmt['original_text'],
                    "annotations": evidence_annotations
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
