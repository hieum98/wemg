"""Common utility functions used across the codebase."""
from typing import Dict, List, Tuple, Optional, Any


def merge_logs(*logs: Dict[str, List[Tuple[str, str]]]) -> Dict[str, List[Tuple[str, str]]]:
    """Merge multiple log dictionaries into one.
    
    Args:
        *logs: Variable number of log dictionaries where each has 
               role names as keys and lists of (input, output) tuples as values.
               
    Returns:
        Merged dictionary with all logs combined.
    """
    if not logs:
        return {}
    
    all_keys = set()
    for log in logs:
        if log:
            all_keys.update(log.keys())
    
    return {
        key: sum((log.get(key, []) for log in logs if log), [])
        for key in all_keys
    }


def log_to_interaction_memory(
    interaction_memory: Any,
    log_data: Dict[str, List[Tuple[str, str]]],
    batch_size: int = 32
) -> None:
    """Log data to interaction memory if available.
    
    Args:
        interaction_memory: The interaction memory instance (or None).
        log_data: Dictionary of role -> list of (input, output) tuples.
        batch_size: Number of entries to process per batch to avoid OOM (default: 100).
                   Large batches are automatically split by log_turn.
    """
    if not interaction_memory or not log_data:
        return
    
    for role, entries in log_data.items():
        if entries:
            inputs, outputs = zip(*entries)
            interaction_memory.log_turn(
                role=role,
                user_input=list(inputs),
                assistant_output=list(outputs),
                batch_size=batch_size
            )

