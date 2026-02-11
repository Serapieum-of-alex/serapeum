"""Utility functions for prompt formatting and variable extraction."""

import re
from typing import Dict, List, Optional

from serapeum.core.base.llms.base import BaseLLM
from serapeum.core.base.llms.types import ChunkType, TextChunk


class SafeFormatter:
    """Safe string formatter that does not raise KeyError if key is missing."""

    def __init__(self, format_dict: Optional[Dict[str, str]] = None):
        """Initialize SafeFormatter with an optional format dictionary."""
        self.format_dict = format_dict or {}

    def format(self, format_string: str) -> str:
        """Format a string, leaving unknown keys unchanged."""
        return re.sub(r"{([^{}]+)}", self._replace_match, format_string)

    def parse(self, format_string: str) -> List[str]:
        """Extract variable names from a format string."""
        return re.findall(
            r"{([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)}", format_string
        )

    def _replace_match(self, match: re.Match) -> str:
        key = match.group(1)
        return str(self.format_dict.get(key, match.group(0)))


def format_string(string_to_format: str, **kwargs: str) -> str:
    """Format a string with kwargs."""
    formatter = SafeFormatter(format_dict=kwargs)
    return formatter.format(string_to_format)


def format_content_blocks(
    content_blocks: List[ChunkType], **kwargs: str
) -> List[ChunkType]:
    """Format content chunks with kwargs."""
    formatter = SafeFormatter(format_dict=kwargs)
    formatted_blocks: List[ChunkType] = []
    for block in content_blocks:
        if isinstance(block, TextChunk):
            # Use the correct attribute for TextChunk (content)
            formatted_blocks.append(TextChunk(content=formatter.format(block.content)))
        else:
            formatted_blocks.append(block)

    return formatted_blocks


def get_template_vars(template_str: str) -> List[str]:
    """Get template variables from a template string."""
    variables = []
    formatter = SafeFormatter()

    for variable_name in formatter.parse(template_str):
        if variable_name:
            variables.append(variable_name)

    return variables
