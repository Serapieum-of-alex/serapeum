"""Helpers for extracting and validating structured data from LLM text.

This module provides utilities to parse JSON/code blocks embedded in model
outputs and a small exception class used by output parsers.
"""

import contextlib
import json
import re
from typing import Any, List

with contextlib.suppress(ImportError):
    import yaml


def _marshal_llm_to_json(output: str) -> str:
    """Extract a substring containing a JSON object or array from a string."""
    output = output.strip()

    left_square = output.find("[")
    left_brace = output.find("{")

    if (left_square < left_brace and left_square != -1) or left_brace == -1:
        left = left_square
        right = output.rfind("]")
    else:
        left = left_brace
        right = output.rfind("}")

    return output[left : right + 1]


def parse_json_markdown(text: str) -> Any:
    """Parse a JSON object/array embedded in fenced markdown.

    If the text contains a fenced block marked as JSON (```json), the content
    of that block is parsed. Otherwise, the function attempts to extract the
    first JSON object/array substring and deserialize it.
    """
    if "```json" in text:
        text = text.split("```json", 1)[1].strip()
        # Remove matching triple backticks without using multi-char strip
        while text.startswith("```"):
            text = text.removeprefix("```").lstrip()
        while text.endswith("```"):
            text = text.removesuffix("```").rstrip()

    json_string = _marshal_llm_to_json(text)

    try:
        json_obj = json.loads(json_string)
    except json.JSONDecodeError as e_json:
        try:
            # Try a lenient YAML parser for slightly invalid JSON
            json_obj = yaml.safe_load(json_string)
        except yaml.YAMLError as e_yaml:
            raise OutputParserException(
                f"Got invalid JSON object. Error: {e_json} {e_yaml}. "
                f"Got JSON string: {json_string}"
            )
        except NameError as exc:
            raise ImportError("Please pip install PyYAML.") from exc

    return json_obj


def parse_code_markdown(text: str, only_last: bool) -> List[str]:
    """Extract code blocks from fenced markdown.

    Args:
        text: The source string that may contain fenced code blocks.
        only_last: When True, return only the last code block; otherwise all.
    """
    pattern = r"```(.*?)```"

    # Remove explicit language tag for python blocks to keep output clean
    python_str_pattern = re.compile(r"^```python", re.IGNORECASE)
    text = python_str_pattern.sub("```", text)

    matches = re.findall(pattern, text, re.DOTALL)
    code = matches[-1] if matches and only_last else matches

    if not code:
        candidate = text.strip()

        if candidate.startswith('"') and candidate.endswith('"'):
            candidate = candidate[1:-1]
        if candidate.startswith("'") and candidate.endswith("'"):
            candidate = candidate[1:-1]
        if candidate.startswith("`") and candidate.endswith("`"):
            candidate = candidate[1:-1]

        if candidate.startswith("```"):
            candidate = re.sub(r"^```[a-zA-Z]*", "", candidate)
        if candidate.endswith("```"):
            candidate = candidate[:-3]
        code = [candidate.strip()]

    return code


def extract_json_str(text: str) -> str:
    """Extract the first JSON object substring from text."""
    match = re.search(r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract json string from output: {text}")

    return match.group()


class OutputParserException(Exception):
    """Raised when an LLM output cannot be parsed into the expected format."""
