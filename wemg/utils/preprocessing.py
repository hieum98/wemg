from typing import List, Dict, Union, Any
import pydantic
import tiktoken


def approximate_token_count(messages: Union[List[Dict[str, str]], str]) -> int:
    """Approximate the number of tokens in a given text using tiktoken."""
    encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(messages, str):
        return len(encoding.encode(messages))
    else:
        token_counts = [len(encoding.encode(m["content"])) for m in messages]
        return sum(token_counts)


def get_node_id(entity: Any) -> str:
    # Lazy imports to avoid circular dependencies
    from wemg.agents import roles
    from wemg.agents.tools import wikidata
    
    if isinstance(entity, wikidata.WikidataEntity):
        return entity.qid
    elif isinstance(entity, roles.open_ie.Entity):
        return str(entity.name)
    else:
        return str(entity)
    

def format_context(memory: str = None, retrieval_info: List[str] = [], reasoning_trace: str = None) -> str:
    retrieval_info = [f"- [Retrieval]: {content.strip()}" for content in retrieval_info]
    retrieval_info = "\n".join(retrieval_info)

    fact_context = [memory, retrieval_info]
    fact_context = "\n".join([fc for fc in fact_context if fc])
    fact_context = "**Information** \n" + fact_context if fact_context else ""
    reasoning_context = f"**Reasoning Trace** \n{reasoning_trace}" if reasoning_trace else ""
    all_context = [fact_context, reasoning_context]
    return "\n---------------\n".join([ac for ac in all_context if ac])
