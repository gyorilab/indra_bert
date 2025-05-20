import json
import logging
from pathlib import Path
from tqdm import tqdm
import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default level
handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def extract_agents_from_statement(statement):
    agents = {}

    def recurse(obj, parent_key=""):
        if isinstance(obj, dict):
            if "db_refs" in obj:
                agents[parent_key] = obj
            else:
                for k, v in obj.items():
                    recurse(v, k)
        elif isinstance(obj, list):
            agent_list = []
            for item in obj:
                if isinstance(item, dict) and "db_refs" in item:
                    agent_list.append(item)
            if agent_list:
                agents[parent_key] = agent_list

    recurse(statement)
    return agents


def grounding_matches(agent_db_refs, raw_grounding):
    for k, v in raw_grounding.items():
        if k == "TEXT":
            continue
        if k in agent_db_refs and agent_db_refs[k] == v:
            return True
    return False


def annotate_text(text, coords, raw_texts, raw_groundings, agent_roles):
    if not coords or not raw_texts or not raw_groundings:
        logger.debug("Empty coords, raw_texts, or raw_groundings.")
        return text, False

    for start, end in coords:
        if text[start:end] not in raw_texts:
            logger.debug(f"Span '{text[start:end]}' not in raw_texts.")
            return text, False

    for raw in raw_texts:
        if not any(raw == rg.get("TEXT", "") for rg in raw_groundings):
            logger.debug(f"Raw text '{raw}' not in any raw_grounding TEXT.")
            return text, False

    for rg in raw_groundings:
        found = False
        for role, val in agent_roles.items():
            vals = val if isinstance(val, list) else [val]
            for agent in vals:
                if grounding_matches(agent.get("db_refs", {}), rg):
                    found = True
                    break
            if found:
                break
        if not found:
            logger.debug(f"No grounding match found for raw_grounding: {rg}")
            return text, False

    spans = [(start, end, text[start:end]) for start, end in coords]
    spans.sort(key=lambda x: -x[0])

    for i in range(len(spans) - 1):
        if spans[i + 1][1] > spans[i][0]:
            logger.debug("Overlapping spans detected.")
            return text, False

    annotated = text
    for start, end, mention in spans:
        role = None
        for r, val in agent_roles.items():
            vals = val if isinstance(val, list) else [val]
            for i, agent in enumerate(vals):
                idx = len(vals) - 1 - i
                if agent.get("db_refs", {}).get("TEXT", "") == mention:
                    role = f"{r}.{idx}" if isinstance(val, list) else r
                    break
            if role:
                break

        if not role:
            logger.debug(f"No role match for mention: {mention}")
            return text, False

        tagged = f"<{role}>{mention}</{role}>"
        annotated = annotated[:start] + tagged + annotated[end:]

    return annotated, True


def process_statement(statement):
    processed = []
    stmt_type = statement.get("type", "")
    agent_roles = extract_agents_from_statement(statement)

    for ev in statement.get("evidence", []):
        try:
            # logger.info(f"Processing evidence: {ev.get('source_hash', '')}")
            text = ev.get("text", "")
            agents_anno = ev.get("annotations", {}).get("agents", {})
            coords = agents_anno.get("coords", [])
            raw_texts = agents_anno.get("raw_text", [])
            raw_groundings = agents_anno.get("raw_grounding", [])

            annotated_text, success = annotate_text(text, coords, raw_texts, raw_groundings, agent_roles)
            if not success:
                continue

            stmt_info = {"type": stmt_type}
            for role, val in agent_roles.items():
                stmt_info[role] = val

            processed.append({
                "matches_hash": statement.get("matches_hash", ""),
                "source_hash": ev.get("source_hash", ""),
                "text": text,
                "annotated_text": annotated_text,
                "statement": stmt_info
            })
        except Exception as e:
            logger.info(f"From statement: {statement.get('matches_hash', '')}") 
            logger.info(f"  From evidence: {ev.get('source_hash', '')}")
            logger.info(f"      Exception: {e}")
            continue

    return processed


def generate_annotated_statements(corpus):
    for statement in tqdm(corpus, desc="Annotating statements"):
        try:
            # logger.info(f"Processing statement: {statement.get('matches_hash', '')}")
            statement_annotations = process_statement(statement)
            for annotated in statement_annotations:
                yield annotated
        except Exception as e:
            logger.info(f"Error processing statement: {statement.get('matches_hash', '')}")
            logger.info(f"Exception: {e}")
            logger.info("Skipping this statement.")
            continue


def main(args):
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Loading input file: {input_path}")
    with open(input_path, "r") as f_in:
        corpus = json.load(f_in)

    logger.info(f"Processing {len(corpus)} statements...")
    with open(output_path, "w") as f_out:
        for annotated_stmt in generate_annotated_statements(corpus):
            f_out.write(json.dumps(annotated_stmt) + "\n")

    logger.info(f"Finished streaming and saving annotations to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process INDRA statements to annotated form")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed debug logging")
    args = parser.parse_args()
    main(args)
