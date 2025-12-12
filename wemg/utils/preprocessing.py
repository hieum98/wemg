from typing import List, Dict, Union
import tiktoken


def approximate_token_count(messages: Union[List[Dict[str, str]], str]) -> int:
    """Approximate the number of tokens in a given text using tiktoken."""
    encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(messages, str):
        return len(encoding.encode(messages))
    else:
        token_counts = [len(encoding.encode(m["content"])) for m in messages]
        return sum(token_counts)
    