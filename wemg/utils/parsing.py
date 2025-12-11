import re
import json
import logging
from typing import List, Union, Dict, Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def extract_info_from_text(text: str, keys: List[str], value_type: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Extract structured information from text that may contain JSON or JSON-like content.
    
    This function attempts multiple strategies to parse the text:
    1. Direct JSON parsing if the text is valid JSON
    2. Finding and parsing JSON objects within the text
    3. Regex-based extraction for partial or malformed JSON
    
    Args:
        text: The text to extract information from
        keys: List of field names to extract
        value_type: List of type hints for each key ('str', 'int', 'float', 'bool', 'list', 'Literal')
                   If None, all keys are treated as strings
    
    Returns:
        Dictionary mapping keys to their extracted values
    """
    if value_type is None:
        value_type = ['str'] * len(keys)
    
    if len(keys) != len(value_type):
        raise ValueError(f"keys and value_type must have the same length. Got {len(keys)} keys and {len(value_type)} types.")
    
    extracted_info = {}
    
    # Strategy 1: Try to parse the entire text as JSON
    try:
        parsed_json = json.loads(text)
        if isinstance(parsed_json, dict):
            # Successfully parsed as JSON, extract values with type conversion
            for key, vtype in zip(keys, value_type):
                if key in parsed_json:
                    extracted_info[key] = _convert_value(parsed_json[key], vtype)
                else:
                    extracted_info[key] = _get_default_value(vtype)
            return extracted_info
    except (json.JSONDecodeError, ValueError):
        pass  # Not valid JSON, try other strategies
    
    # Strategy 2: Try to find and parse JSON objects within the text
    # Look for patterns like {...} that might be JSON
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    json_matches = re.finditer(json_pattern, text, re.DOTALL)
    
    for match in json_matches:
        try:
            potential_json = match.group(0)
            parsed_json = json.loads(potential_json)
            if isinstance(parsed_json, dict):
                # Check if this JSON object contains any of our keys
                if any(key in parsed_json for key in keys):
                    for key, vtype in zip(keys, value_type):
                        if key in parsed_json and key not in extracted_info:
                            extracted_info[key] = _convert_value(parsed_json[key], vtype)
        except (json.JSONDecodeError, ValueError):
            continue
    
    # Strategy 3: Regex-based extraction for each field
    # This handles partial or malformed JSON
    for key, vtype in zip(keys, value_type):
        if key in extracted_info:
            continue  # Already extracted from JSON
        
        value = _extract_field_with_regex(text, key, vtype)
        extracted_info[key] = value
    
    return extracted_info


def _convert_value(value: Any, vtype: str) -> Any:
    """Convert a value to the specified type."""
    try:
        if vtype in ['str', 'Literal']:
            return str(value)
        elif vtype == 'bool':
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ('true', 'yes', '1')
            return bool(value)
        elif vtype == 'int':
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str):
                # Handle numeric strings, including those with decimals
                return int(float(value))
            return int(value)
        elif vtype == 'float':
            return float(value)
        elif vtype in ['list', 'List']:
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                # Try to parse as JSON array first
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return parsed
                except:
                    pass
                # Split by comma if it's a comma-separated string
                return [item.strip().strip('"\'') for item in value.split(',') if item.strip()]
            return [value]
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to convert value '{value}' to type '{vtype}': {e}")
        return _get_default_value(vtype)
    
    return _get_default_value(vtype)


def _get_default_value(vtype: str) -> Any:
    """Get the default value for a given type."""
    if vtype in ['str', 'Literal']:
        return ""
    elif vtype == 'bool':
        return False
    elif vtype in ['int', 'float']:
        return 0
    elif vtype in ['list', 'List']:
        return []
    else:
        return None


def _extract_field_with_regex(text: str, key: str, vtype: str) -> Any:
    """Extract a field value using regex patterns."""
    
    if vtype in ['str', 'Literal']:
        # Try multiple string patterns
        patterns = [
            rf'"{key}":\s*"([^"]*)"',  # "key": "value"
            rf"'{key}':\s*'([^']*)'",  # 'key': 'value'
            rf'{key}:\s*"([^"]*)"',    # key: "value" (no quotes on key)
            rf'{key}\s+(?:level|score|rating|value)?\s*is\s+"([^"]*)"',  # The key [level|score] is "value"
            rf'{key}\s+(?:level|score|rating|value)?\s*is\s+([^\n,\.]+)',  # The key is value (no quotes)
            rf'{key}:\s*([^\n,\}}]+)',  # key: value (no quotes)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Clean up common artifacts and quotes
                value = value.rstrip(',').strip()
                value = value.strip('"\'')
                return value
        
        return ""
    
    elif vtype == 'bool':
        # Match boolean values
        patterns = [
            rf'"{key}":\s*(true|false|True|False|TRUE|FALSE)',
            rf'{key}:\s*(true|false|True|False|TRUE|FALSE)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).lower() in ('true', '1', 'yes')
        
        return False
    
    elif vtype in ['int', 'float']:
        # Match numeric values
        patterns = [
            rf'"{key}":\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
            rf"'{key}':\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",  # 'key': value
            rf'{key}:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
            rf'{key}\s+(?:level|score|rating|value)?\s*is\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',  # "The key is 5"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    if vtype == 'int':
                        return int(float(match.group(1)))
                    else:
                        return float(match.group(1))
                except ValueError:
                    pass
        
        return 0
    
    elif vtype in ['list', 'List']:
        # Match array/list patterns
        patterns = [
            rf'"{key}":\s*\[(.*?)\]',  # Complete array
            rf'{key}:\s*\[(.*?)\]',     # Array without quotes on key
            rf'"{key}":\s*\[(.*)',      # Incomplete array (missing closing bracket)
            rf'{key}:\s*\[(.*)',        # Incomplete array without quotes
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                content = match.group(1).strip()
                
                # Try to parse as JSON array first
                try:
                    # Add closing bracket if missing and try to parse
                    if not content.endswith(']'):
                        potential_json = '[' + content + ']'
                    else:
                        potential_json = '[' + content
                    
                    parsed = json.loads(potential_json)
                    if isinstance(parsed, list):
                        return parsed
                except:
                    pass
                
                # Fallback: split by comma or newline
                # Handle both comma-separated and newline-separated items
                list_items = re.split(r',\s*\n|\n|,', content)
                list_items = [
                    item.strip().strip('"\'').strip()
                    for item in list_items
                    if item.strip() and item.strip() not in ('', '}', '{')
                ]
                
                if list_items:
                    return list_items
        
        return []
    
    else:
        raise ValueError(f"Unsupported value type: {vtype}. Supported types are: str, bool, int, float, list.")
    
