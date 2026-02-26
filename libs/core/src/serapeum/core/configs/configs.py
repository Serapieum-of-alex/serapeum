"""Configuration helpers and runtime settings for Serapeum."""

from dataclasses import dataclass
from typing import Optional

from serapeum.core.llms import LLM
from serapeum.core.types import StructuredOutputMode


@dataclass
class _Configs:
    """Configs for the Llama Index, lazily initialized."""

    _llm: Optional[LLM] = None

    @property
    def llm(self) -> LLM:
        """Get the LLM."""
        return self._llm

    @llm.setter
    def llm(self, llm) -> None:
        """Set the LLM."""
        self._llm = llm

    @property
    def structured_output_mode(self) -> StructuredOutputMode:
        """Get the pydantic program mode."""
        return self.llm.structured_output_mode

    @structured_output_mode.setter
    def structured_output_mode(self, structured_output_mode: StructuredOutputMode) -> None:
        """Set the pydantic program mode."""
        self.llm.structured_output_mode = structured_output_mode


# Singleton
Configs = _Configs()
