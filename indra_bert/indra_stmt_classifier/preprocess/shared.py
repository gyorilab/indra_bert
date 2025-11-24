import re

ENTITY_OPEN = "<e>"
ENTITY_CLOSE = "</e>"
ENTITY_TAG_PATTERN = re.compile(r"<e>(.*?)</e>", flags=re.IGNORECASE)

BINARY_LABEL_MAPPING = {"no_relation": 0, "has_relation": 1}
IGNORE_INDEX = -100

def normalize_entity_tags(text: str) -> str:
    text = re.sub(r"<[^/][^>]*>", ENTITY_OPEN, text)
    text = re.sub(r"</[^>]+>", ENTITY_CLOSE, text)
    return text
