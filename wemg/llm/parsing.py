import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def extract_info_from_text(
    text: str,
    keys: List[str],
    value_type: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if value_type is None:
        value_type = ["str"] * len(keys)

    if len(keys) != len(value_type):
        raise ValueError(
            f"keys and value_type must have the same length. "
            f"Got {len(keys)} keys and {len(value_type)} types."
        )

    extracted_info: Dict[str, Any] = {}

    # Strategy 1: direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            for key, vtype in zip(keys, value_type):
                if key in parsed:
                    extracted_info[key] = _convert_value(parsed[key], vtype)
                else:
                    extracted_info[key] = _get_default_value(vtype)
            return extracted_info
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: find JSON objects within text
    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    for match in re.finditer(json_pattern, text, re.DOTALL):
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and any(k in parsed for k in keys):
                for key, vtype in zip(keys, value_type):
                    if key in parsed and key not in extracted_info:
                        extracted_info[key] = _convert_value(parsed[key], vtype)
        except (json.JSONDecodeError, ValueError):
            continue

    # Strategy 3: per-field regex extraction
    for key, vtype in zip(keys, value_type):
        if key not in extracted_info:
            extracted_info[key] = _extract_field_with_regex(text, key, vtype)

    return extracted_info


def _convert_value(value: Any, vtype: str) -> Any:
    try:
        if vtype in ("str", "Literal"):
            return str(value)

        if vtype == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("true", "yes", "1")
            return bool(value)

        if vtype == "int":
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str):
                return int(float(value))
            return int(value)

        if vtype == "float":
            return float(value)

        if vtype in ("list", "List"):
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    pass
                return [
                    item.strip().strip("\"'")
                    for item in value.split(",")
                    if item.strip()
                ]
            return [value]

    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to convert value '{value}' to type '{vtype}': {e}")
        return _get_default_value(vtype)

    return _get_default_value(vtype)


def _get_default_value(vtype: str) -> Any:
    defaults = {
        "str": "",
        "Literal": "",
        "bool": False,
        "int": 0,
        "float": 0,
        "list": [],
        "List": [],
    }
    return defaults.get(vtype)


def _extract_field_with_regex(text: str, key: str, vtype: str) -> Any:
    if vtype in ("str", "Literal"):
        patterns = [
            rf'"{key}":\s*"([^"]*)"',
            rf"'{key}':\s*'([^']*)'",
            rf'{key}:\s*"([^"]*)"',
            rf'{key}\s+(?:level|score|rating|value)?\s*is\s+"([^"]*)"',
            rf"{key}\s+(?:level|score|rating|value)?\s*is\s+([^\n,.]+)",
            rf"{key}:\s*([^\n,}}]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                value = match.group(1).strip().rstrip(",").strip().strip("\"'")
                return value
        return ""

    if vtype == "bool":
        patterns = [
            rf'"{key}":\s*(true|false|True|False|TRUE|FALSE)',
            rf"{key}:\s*(true|false|True|False|TRUE|FALSE)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).lower() in ("true", "1", "yes")
        return False

    if vtype in ("int", "float"):
        num = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        patterns = [
            rf'"{key}":\s*{num}',
            rf"'{key}':\s*{num}",
            rf"{key}:\s*{num}",
            rf"{key}\s+(?:level|score|rating|value)?\s*is\s+{num}",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    if vtype == "int":
                        return int(float(match.group(1)))
                    return float(match.group(1))
                except ValueError:
                    pass
        return 0

    if vtype in ("list", "List"):
        patterns = [
            rf'"{key}":\s*\[(.*?)\]',
            rf"{key}:\s*\[(.*?)\]",
            rf'"{key}":\s*\[(.*)',
            rf"{key}:\s*\[(.*)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                content = match.group(1).strip()
                try:
                    if not content.endswith("]"):
                        potential = "[" + content + "]"
                    else:
                        potential = "[" + content
                    parsed = json.loads(potential)
                    if isinstance(parsed, list):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    pass

                items = re.split(r",\s*\n|\n|,", content)
                items = [
                    item.strip().strip("\"'").strip()
                    for item in items
                    if item.strip() and item.strip() not in ("", "}", "{")
                ]
                if items:
                    return items
        return []

    raise ValueError(
        f"Unsupported value type: {vtype}. "
        f"Supported types are: str, bool, int, float, list."
    )
