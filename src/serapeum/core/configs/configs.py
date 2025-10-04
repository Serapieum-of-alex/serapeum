from dataclasses import dataclass
from typing import Optional


from serapeum.core.llm import LLM
from serapeum.core.models import PydanticProgramMode


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
    def pydantic_program_mode(self) -> PydanticProgramMode:
        """Get the pydantic program mode."""
        return self.llm.pydantic_program_mode

    @pydantic_program_mode.setter
    def pydantic_program_mode(self, pydantic_program_mode: PydanticProgramMode) -> None:
        """Set the pydantic program mode."""
        self.llm.pydantic_program_mode = pydantic_program_mode



# Singleton
Configs = _Configs()
