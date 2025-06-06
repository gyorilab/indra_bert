import json
from pathlib import Path

biored_train = json.load(open(Path.cwd()/'data'/'ner_data'/'BioRED'/'Train.BioC.JSON'))

abstracts = []
for doc in biored_train['documents']:
    title = doc['passages'][0]['text']
    paragraph = doc['passages'][1]['text']
    abstract = title + ' ' + paragraph
    abstracts.append(abstract)

from indra.sources.indra_bert.processor import IndraBertProcessor
from indra_bert import IndraStructuredExtractor
from tqdm import tqdm
import pickle

stmts = []
ise = IndraStructuredExtractor(
    ner_model_path="thomaslim6793/indra_bert_ner_agent_detection",
    stmt_model_path="thomaslim6793/indra_bert_indra_stmt_classifier_pubmedbert",
    role_model_path="thomaslim6793/indra_bert_indra_stmt_agents_role_assigner",
    stmt_conf_threshold=0.9
)
for abstract in tqdm(abstracts, desc="Processing abstracts"):
    res = ise.extract_structured_statements(abstract)
    ip = IndraBertProcessor(res)
    stmts.extend(ip.statements)

cur_dir = Path.cwd()
pickle.dump(stmts, open(cur_dir/'BioRED_train_statements.pkl', 'wb'))
