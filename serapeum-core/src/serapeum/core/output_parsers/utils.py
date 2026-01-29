"""Helpers for extracting and validating structured data from LLM text.

This module provides utilities to parse JSON/code blocks embedded in model
outputs and a small exception class used by output parsers.
"""

import contextlib
import json
import re
from typing import Any, List

from serapeum.core.utils.schemas import marshal_llm_to_json

__all__ = ["OutputParserException", "parse_json_markdown", "parse_code_markdown"]

with contextlib.suppress(ImportError):
    import yaml


class OutputParserException(Exception):
    """Exception raised for errors encountered during output parsing."""

    pass


def parse_json_markdown(text: str) -> Any:
    r"""Parse a JSON object/array embedded in fenced markdown.

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

    json_string = marshal_llm_to_json(text)

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
    r"""Extract code blocks from fenced markdown.

    Args:
        text (str): The markdown text to parse.
        only_last (bool): If True, return only the last code block.

    Returns:
        List[str]: List of code block contents.
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
