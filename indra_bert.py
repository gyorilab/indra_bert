from itertools import combinations
from ner_agent_detector.model import AgentNERModel
from indra_stmt_classifier.model import IndraStmtClassifier
from indra_agent_role_assigner.model import IndraAgentsTagger
from utils.annotate import annotate_entities

class IndraStructuredExtractor:
    def __init__(self, ner_model_path, stmt_model_path, role_model_path, stmt_conf_threshold=0.95):
        self.ner_model = AgentNERModel(ner_model_path)
        self.stmt_model = IndraStmtClassifier(stmt_model_path)
        self.role_model = IndraAgentsTagger(role_model_path)
        self.stmt_conf_threshold = stmt_conf_threshold

    def get_entity_pairs(self, entity_preds):
        return list(combinations(entity_preds['entity_spans'], 2))

    def extract_structured_statements(self, text):
        stmts = []
        entity_preds = self.ner_model.predict(text)
        pairs = self.get_entity_pairs(entity_preds)

        for pair in pairs:
            annotated_text = annotate_entities(text, pair)
            stmt_pred = self.stmt_model.predict(annotated_text)
            stmt_conf = stmt_pred.get('confidence', 0.0)

            if stmt_conf < self.stmt_conf_threshold:
                continue

            role_pred = self.role_model.predict(stmt_pred['predicted_label'], annotated_text)

            stmt = {
                'original_text': text,
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
                }
            }

            stmts.append(stmt)

        return stmts

    def get_json_indra_stmts(self, text, source_api="indra_bert"):
        """Extract statements and convert to INDRA-style JSON with agent coords."""
        structured_statements = self.extract_structured_statements(text)
        indra_statements = []

        for stmt in structured_statements:
            stmt_type = stmt['stmt_pred']['label']
            confidence = stmt['stmt_pred']['confidence']
            roles = stmt['role_pred']['roles']

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

            indra_stmt = {
                "type": stmt_type,
                **agent_roles,
                "belief": confidence,
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
