from typing import List, Dict, Union
import tiktoken
from traitlets import Any

from wemg.agents import roles
from wemg.agents.tools import wikidata


def approximate_token_count(messages: Union[List[Dict[str, str]], str]) -> int:
    """Approximate the number of tokens in a given text using tiktoken."""
    encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(messages, str):
        return len(encoding.encode(messages))
    else:
        token_counts = [len(encoding.encode(m["content"])) for m in messages]
        return sum(token_counts)


def get_node_id(entity: Any) -> str:
    if isinstance(entity, wikidata.WikidataEntity):
        return entity.qid
    elif isinstance(entity, roles.open_ie.Entity):
        return str(entity.name)
    else:
        return str(entity)
