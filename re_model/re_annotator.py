# re_model/re_annotator.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict
import re


class REAnnotator:
    def __init__(self, relation_model_path: str, agent_role_model_path: str):
        # Load relation classification model
        self.relation_tokenizer = AutoTokenizer.from_pretrained(relation_model_path)
        self.relation_model = AutoModelForSequenceClassification.from_pretrained(relation_model_path)
        self.relation_model.eval()

        # Load agent role classification model
        self.role_tokenizer = AutoTokenizer.from_pretrained(agent_role_model_path)
        self.role_model = AutoModelForSequenceClassification.from_pretrained(agent_role_model_path)
        self.role_model.eval()

        # Relation label mapping (you can load this from json or config if needed)
        self.id2relation = {0: "Acetylation", 1: "Activation", 2: "Inhibition"}  # example
        # Agent role mapping (you can load this too)
        self.id2role = {0: "enz", 1: "sub", 2: "subj", 3: "obj"}  # example

    def annotate(self, text: str) -> Dict:
        """
        Annotate a text to predict relation and agent roles.
        """

        # === Step 1: Extract agent spans ===
        agent_spans = self._extract_agents(text)

        # === Step 2: Predict relation type ===
        relation_type = self._predict_relation(text)

        # === Step 3: Predict roles for agents ===
        agent_roles = []
        for agent in agent_spans:
            role = self._predict_agent_role(text, agent, relation_type)
            agent_roles.append({
                "text": agent,
                "role": role
            })

        # === Step 4: Build structured result ===
        result = {
            "relation": relation_type,
            "agents": agent_roles
        }

        return result

    def _extract_agents(self, text: str) -> List[str]:
        """
        Extract agent mentions from text in <Agent> ... </Agent> tags
        """
        pattern = r"<Agent>(.*?)</Agent>"
        agents = re.findall(pattern, text)
        return agents

    def _predict_relation(self, text: str) -> str:
        """
        Predict relation type from the text
        """
        encoding = self.relation_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.relation_model(**encoding)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        return self.id2relation[prediction]

    def _predict_agent_role(self, text: str, agent: str, relation_type: str) -> str:
        """
        Predict the role of the agent based on relation type + context
        """
        # Simple format: "<relation> | <agent> | <context>"
        input_text = f"{relation_type} | {agent} | {text}"
        encoding = self.role_tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.role_model(**encoding)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        return self.id2role[prediction]

    def annotate_pretty(self, text: str) -> str:
        """
        Same as annotate, but returns text with in-place role annotations.
        """
        result = self.annotate(text)

        annotated_text = text
        for agent in result["agents"]:
            role = agent["role"]
            pattern = r"<Agent>(.*?)</Agent>"

            # Only replace the first occurrence
            annotated_text = re.sub(pattern, f"<Agent {role}>\\1</Agent {role}>", annotated_text, count=1)

        return annotated_text


if __name__ == "__main__":
    # Example usage
    annotator = REAnnotator("re_model/relation_classifier", "re_model/agent_role_assignment")

    text = "beta (2) <Agent>AR</Agent> stimulation resulted in cyclic AMP dependent inhibition of <Agent>histone</Agent> deacetylase-8 (HDAC8) activity and increased <Agent>histone</Agent> acetylation."

    result = annotator.annotate(text)
    print("Structured Result:")
    print(result)

    pretty = annotator.annotate_pretty(text)
    print("\nAnnotated Text:")
    print(pretty)
