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


class JsonParser:
    """Utility class for parsing and fixing malformed JSON from LLM outputs.

    This class handles common JSON formatting issues that arise from LLM responses,
    such as literal newlines in strings, unescaped control characters, and
    improperly escaped quotes.
    """
    def __init__(self, text: str) -> None:
        text = text.strip()
        self.text = text

    @staticmethod
    def parse(json_str: str) -> Any:
        """Parse a JSON string with automatic error recovery.

        Args:
            json_str: Raw JSON string that may contain formatting issues.

        Returns:
            Parsed JSON object (dict, list, etc.).

        Raises:
            ValueError: If the JSON cannot be parsed even after attempting fixes.

        Examples:
            >>> JsonParser.parse('{"name": "test"}')
            {'name': 'test'}

            >>> # Handles literal newlines in strings
            >>> JsonParser.parse('{"text": "line1\\nline2"}')
            {'text': 'line1\\nline2'}
        """
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try to fix common issues and retry
            try:
                fixed_json = JsonParser.fix_json_string(json_str)
                return json.loads(fixed_json)
            except json.JSONDecodeError as retry_error:
                raise ValueError(
                    f"Failed to parse JSON from LLM output. "
                    f"Original error: {e}. Retry error: {retry_error}. "
                    f"JSON string (first 500 chars): {json_str[:500]}"
                )

    def extract_str(self) -> str:
        """Extract the first JSON object substring from text."""

        json_str = _marshal_llm_to_json(self.text)

        # Validate it's actually valid JSON
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            # Fall back to regex with non-greedy matching
            match = re.search(r"\{.*?\}", self.text, re.MULTILINE | re.IGNORECASE | re.DOTALL)
            if not match:
                raise ValueError(f"Could not extract json string from output: {self.text}")

            return match.group()

    @staticmethod
    def fix_json_string(json_str: str) -> str:
        """Fix common JSON formatting issues from LLM outputs.

        Handles:
        - Single-escaped quotes (\\') that should be unescaped
        - Literal newlines inside string values
        - Literal carriage returns and tabs
        - Other control characters (converted to unicode escapes)

        Args:
            json_str: Raw JSON string with potential formatting issues.

        Returns:
            Fixed JSON string that should be valid JSON.

        Examples:
            >>> JsonParser.fix_json_string('{"key": "value"}')
            '{"key": "value"}'

            >>> # Fixes literal newlines
            >>> result = JsonParser.fix_json_string('{"text": "line1\\nline2"}')
            >>> '\\\\n' in result
            True
        """
        # Replace single-escaped quotes that should be unescaped
        json_str = json_str.replace(r"\'", "'")

        # Fix literal newlines/control chars inside JSON strings
        result = []
        in_string = False
        escape_next = False

        for char in json_str:
            if escape_next:
                result.append(char)
                escape_next = False
                continue

            if char == '\\':
                result.append(char)
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                result.append(char)
                continue

            # If we're inside a string, escape control characters
            if in_string:
                if char == '\n':
                    result.append('\\n')
                elif char == '\r':
                    result.append('\\r')
                elif char == '\t':
                    result.append('\\t')
                elif ord(char) < 32:
                    # Escape other control characters as unicode
                    result.append(f'\\u{ord(char):04x}')
                else:
                    result.append(char)
            else:
                result.append(char)

        return ''.join(result)


class OutputParserException(Exception):
    """Raised when an LLM output cannot be parsed into the expected format."""
