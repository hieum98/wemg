import json
import logging
import re
import types
from typing import Annotated, Any, Dict, List, Literal, Optional, Union, get_args, get_origin

logger = logging.getLogger(__name__)


def extraction_type_from_annotation(annotation: Any) -> tuple[str, bool]:
    """Map a Pydantic field annotation to (value_type, is_optional) for extract_info_from_text.

    Strips ``Optional`` / ``Union[T, None]`` / ``T | None``, ``Annotated``, and normalizes
    ``list`` / ``List[...]`` to the parser's ``list`` label.
    """
    is_optional = False
    ann: Any = annotation

    while True:
        origin = get_origin(ann)
        args = get_args(ann) or ()

        if origin is Annotated:
            if args:
                ann = args[0]
                continue
            break

        if origin is Union or origin is types.UnionType:
            has_none = any(a is type(None) for a in args)
            non_none = [a for a in args if a is not type(None)]
            if has_none:
                is_optional = True
            if len(non_none) == 1:
                ann = non_none[0]
                continue
            if len(non_none) > 1:
                ann = non_none[0]
                continue
            return "str", is_optional

        if origin is list:
            return "list", is_optional

        if origin is Literal:
            return "Literal", is_optional

        break

    if isinstance(ann, type):
        if ann is str:
            return "str", is_optional
        if ann is int:
            return "int", is_optional
        if ann is float:
            return "float", is_optional
        if ann is bool:
            return "bool", is_optional

    name = getattr(ann, "__name__", "") or "str"
    if name in ("List", "list"):
        return "list", is_optional
    if name in ("str", "int", "float", "bool", "Literal"):
        return name, is_optional

    return name, is_optional


def extract_info_from_text(
    text: str,
    keys: List[str],
    value_type: Optional[List[str]] = None,
    field_optional: Optional[List[bool]] = None,
) -> Dict[str, Any]:
    if value_type is None:
        value_type = ["str"] * len(keys)

    if len(keys) != len(value_type):
        raise ValueError(
            f"keys and value_type must have the same length. "
            f"Got {len(keys)} keys and {len(value_type)} types."
        )

    if field_optional is not None and len(field_optional) != len(keys):
        raise ValueError(
            f"keys and field_optional must have the same length. "
            f"Got {len(keys)} keys and {len(field_optional)} flags."
        )

    optional_by_key = field_optional if field_optional is not None else [False] * len(keys)

    extracted_info: Dict[str, Any] = {}

    # Strategy 1: direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            for key, vtype, opt in zip(keys, value_type, optional_by_key):
                if key in parsed:
                    extracted_info[key] = _convert_value(parsed[key], vtype)
                else:
                    if opt:
                        extracted_info[key] = None
                    else:
                        raise ValueError(f"JSON missing required field: {key}")
            return extracted_info
    except (json.JSONDecodeError, ValueError):
                logger.warning("Direct JSON parse failed; falling back to embedded JSON and regex extraction.")

    # Strategy 2: find JSON objects within text
    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    for match in re.finditer(json_pattern, text, re.DOTALL):
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and any(k in parsed for k in keys):
                for key, vtype, opt in zip(keys, value_type, optional_by_key):
                    if key in parsed and key not in extracted_info:
                        extracted_info[key] = _convert_value(parsed[key], vtype)
        except (json.JSONDecodeError, ValueError):
            continue

    # Strategy 3: per-field regex extraction
    for key, vtype, opt in zip(keys, value_type, optional_by_key):
        if key not in extracted_info:
            extracted_info[key] = _extract_field_with_regex(text, key, vtype, optional=opt)

    return extracted_info


def _convert_value(value: Any, vtype: str) -> Any:
    if value is None:
        return None
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
        raise ValueError(f"Failed to convert value '{value}' to type '{vtype}': {e}")

    raise ValueError(f"Could not convert value '{value}' to type '{vtype}'")


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


def _extract_field_with_regex(
    text: str, key: str, vtype: str, *, optional: bool = False
) -> Any:
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
        if optional:
            return None
        raise ValueError(f"Regex extraction failed for required field: '{key}'")

    if vtype == "bool":
        patterns = [
            rf'"{key}":\s*(true|false|True|False|TRUE|FALSE)',
            rf"{key}:\s*(true|false|True|False|TRUE|FALSE)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).lower() in ("true", "1", "yes")
        if optional:
            return None
        raise ValueError(f"Regex extraction failed for required field: '{key}'")

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
        if optional:
            return None
        raise ValueError(f"Regex extraction failed for required field: '{key}'")

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
        if optional:
            return None
        raise ValueError(f"Regex extraction failed for required field: '{key}'")

    raise ValueError(
        f"Unsupported value type: {vtype}. "
        f"Supported types are: str, bool, int, float, list."
    )
