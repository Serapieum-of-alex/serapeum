"""Utilities for working with JSON schemas and LLMs."""
from __future__ import annotations
import copy
import json
import re
from dataclasses import dataclass
from typing import Any

import yaml

__all__ = [
    "SchemaFormatter",
    "Schema",
    "JsonParser",
    "PYDANTIC_FORMAT_TMPL",
]

PYDANTIC_FORMAT_TMPL = """
Here's a JSON schema to follow strictly:
{schema}

IMPORTANT: Return ONLY a valid JSON object with the actual data, NOT the schema itself.
Do not include "properties", "required", "title", or "type" fields in your response.
Return the data as a JSON object that matches the schema structure.
"""


def marshal_llm_to_json(output: str) -> str:
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


class JsonParser:
    """Utility class for parsing and fixing malformed JSON from LLM outputs.

    This class handles common JSON formatting issues that arise from LLM responses,
    such as literal newlines in strings, unescaped control characters, and
    improperly escaped quotes.
    """

    def __init__(self, text: str) -> None:
        """Initialize JsonParser with the given text."""
        text = text.strip()
        self.text = text

    @staticmethod
    def parse(json_str: str) -> Any:
        r"""Parse a JSON string with automatic error recovery.

        Args:
            json_str: Raw JSON string that may contain formatting issues.

        Returns:
            Parsed JSON object (dict, list, etc.).

        Raises:
            ValueError: If the JSON cannot be parsed even after attempting fixes.

        Examples:
            ```python
            >>> JsonParser.parse('{"name": "test"}')
            {'name': 'test'}

            ```
            - Handles literal newlines in strings
            ```python
            >>> JsonParser.parse('{"text": "line1\\nline2"}')
            {'text': 'line1\\nline2'}

            ```
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
        json_str = marshal_llm_to_json(self.text)

        # Validate it's actually valid JSON
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            # Fall back to regex with non-greedy matching
            match = re.search(
                r"{.*?}", self.text, re.MULTILINE | re.IGNORECASE | re.DOTALL
            )
            if not match:
                raise ValueError(
                    f"Could not extract json string from output: {self.text}"
                )

            return match.group()

    @staticmethod
    def fix_json_string(json_str: str) -> str:
        r"""Fix common JSON formatting issues from LLM outputs.

        Handles:
        - Single-escaped quotes (\') that should be unescaped
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
            >>> '\\n' in result
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

            if char == "\\":
                result.append(char)
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                result.append(char)
                continue

            # If we're inside a string, escape control characters
            if in_string:
                if char == "\n":
                    result.append("\\n")
                elif char == "\r":
                    result.append("\\r")
                elif char == "\t":
                    result.append("\\t")
                elif ord(char) < 32:
                    # Escape other control characters as unicode
                    result.append(f"\\u{ord(char):04x}")
                else:
                    result.append(char)
            else:
                result.append(char)

        return "".join(result)


class SchemaFormatter:
    """Utility class for formatting JSON schemas into LLM-friendly representations.

    This class converts verbose Pydantic JSON schemas into simplified, readable
    formats that are easier for LLMs to understand and follow.
    """

    @staticmethod
    def simplify(schema_dict: dict, excluded_keys: list[str] | None = None) -> str:
        """Create a simplified, example-based schema representation.

        Converts a full JSON Schema dictionary into a cleaner format showing
        field names, types, and requirements without the verbose schema structure.

        Args:
            schema_dict: Full JSON Schema dictionary from Pydantic's model_json_schema().
            excluded_keys: List of schema keys to exclude from the output.

        Returns:
            Simplified schema string in a human-readable format.

        Examples:
            >>> from pydantic import BaseModel
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            >>> schema = Person.model_json_schema()
            >>> print(SchemaFormatter.simplify(schema))
            Expected JSON structure:
            {
              "name": <string> [REQUIRED],
              "age": <integer> [REQUIRED]
            }
        """
        # Create a copy to avoid mutating the original
        schema_dict = dict(schema_dict)

        # Remove excluded keys
        for key in excluded_keys or []:
            schema_dict.pop(key, None)

        properties = schema_dict.get("properties", {})
        required = schema_dict.get("required", [])

        # Build a cleaner format showing field names, types, and requirements
        lines = ["Expected JSON structure:", "{"]

        for i, (field_name, field_info) in enumerate(properties.items()):
            field_type = field_info.get("type", "any")
            field_desc = field_info.get("description", "")
            is_required = field_name in required

            # Format: "field_name": <type> [REQUIRED] - description
            req_marker = " [REQUIRED]" if is_required else " [OPTIONAL]"
            desc_marker = f" - {field_desc}" if field_desc else ""

            comma = "," if i < len(properties) - 1 else ""
            lines.append(
                f'  "{field_name}": <{field_type}>{req_marker}{desc_marker}{comma}'
            )

        lines.append("}")

        return "\n".join(lines)

    @staticmethod
    def format_for_llm(
        schema_dict: dict,
        template: str = PYDANTIC_FORMAT_TMPL,
        excluded_keys: list[str] | None = None,
        escape_json: bool = True,
    ) -> str:
        """Format a schema dictionary for inclusion in an LLM prompt.

        Args:
            schema_dict: Full JSON Schema dictionary.
            template: Template string with {schema} placeholder.
            excluded_keys: List of schema keys to exclude.
            escape_json: Whether to escape JSON braces for prompt templates.

        Returns:
            Formatted string ready to be added to LLM prompts.

        Examples:
            >>> from pydantic import BaseModel
            >>> class Task(BaseModel):
            ...     title: str
            ...     done: bool = False
            >>> schema = Task.model_json_schema()
            >>> prompt = SchemaFormatter.format_for_llm(schema, escape_json=False)
            >>> "Expected JSON structure" in prompt
            True
        """
        simplified_schema = SchemaFormatter.simplify(schema_dict, excluded_keys)
        output_str = template.format(schema=simplified_schema)

        if escape_json:
            return output_str.replace("{", "{{").replace("}", "}}")
        else:
            return output_str


@dataclass
class Schema:
    """Container for resolved and referenced schema variants."""

    full_schema: dict[str, Any]
    resolved_schema: dict[str, Any] | None = None
    referenced_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Post-init docstring."""
        self.resolved_schema = self.resolve_references(inline=True)
        self.referenced_schema = self.resolve_references(inline=False)

    def resolve_references(self, inline: bool = False) -> dict[str, Any]:
        defs = (
            self.full_schema.get("$defs") or self.full_schema.get("definitions") or {}
        )
        # Inline any local references first
        if inline:
            schema = self._resolve_local_refs(copy.deepcopy(self.full_schema), defs)
            keys = ["type", "properties", "required"]
        else:
            schema = self.full_schema
            keys = ["type", "properties", "required", "definitions", "$defs"]

        # Now keep only the keys relevant for tool providers
        parameters = {k: v for k, v in schema.items() if k in keys}
        return parameters

    @staticmethod
    def _resolve_local_refs(obj: Any, defs: dict[str, Any]) -> Any:
        """Recursively inline local $ref objects using the provided defs."""
        if isinstance(obj, dict):
            if "$ref" in obj and isinstance(obj["$ref"], str):
                ref: str = obj["$ref"]
                if ref.startswith("#/$defs/") or ref.startswith("#/definitions/"):
                    name = ref.split("/")[-1]
                    if name in defs:
                        # Deep-copy to avoid mutating the original defs
                        return Schema._resolve_local_refs(
                            copy.deepcopy(defs[name]), defs
                        )
                    # If not found, fall through and return as-is
            # Recurse into mapping
            return {k: Schema._resolve_local_refs(v, defs) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [Schema._resolve_local_refs(v, defs) for v in obj]
        return obj


class MarkdownParserException(Exception):
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
            raise MarkdownParserException(
                f"Got invalid JSON object. Error: {e_json} {e_yaml}. "
                f"Got JSON string: {json_string}"
            )
        except NameError as exc:
            raise ImportError("Please pip install PyYAML.") from exc

    return json_obj


def parse_code_markdown(text: str, only_last: bool) -> list[str]:
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
