import json
from pathlib import Path

from indra.sources.indra_bert.processor import IndraBertProcessor
from indra_bert import IndraStructuredExtractor
from tqdm import tqdm
import pickle

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ner_model_path", required=True)
    parser.add_argument("--stmt_model_path", required=True)
    parser.add_argument("--role_model_path", required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    logger.info("Loading BioRED training data...")
    biored_train = json.load(open(Path.cwd()/'data'/'ner_data'/'BioRED'/'Train.BioC.JSON'))

    logger.info("Loading abstracts from BioRED training data...")
    abstracts = []
    for doc in biored_train['documents']:
        title = doc['passages'][0]['text']
        paragraph = doc['passages'][1]['text']
        abstract = title + ' ' + paragraph
        abstracts.append(abstract)

    logger.info("Loading INDRA structured extractor...")
    ise = IndraStructuredExtractor(
        ner_model_path=args.ner_model_path,
        stmt_model_path=args.stmt_model_path,
        role_model_path=args.role_model_path,
        stmt_conf_threshold=0.9
    )
    logger.info("Extracting structured statements...")
    stmts = []
    for abstract in tqdm(abstracts, desc="Processing abstracts"):
        res = ise.extract_structured_statements(abstract)
        ip = IndraBertProcessor(res)
        stmts.extend(ip.statements)
    logger.info(f"Extracted {len(stmts)} statements.")
    logger.info("Saving statements to pickle file...")
    cur_dir = Path.cwd()
    pickle.dump(stmts, open(cur_dir/'BioRED_train_statements.pkl', 'wb'))

if __name__ == "__main__":
    main()
