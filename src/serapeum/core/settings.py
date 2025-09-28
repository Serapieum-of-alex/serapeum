from dataclasses import dataclass
from typing import Any, Callable, List, Optional


from serapeum.core.llms import LLM
from serapeum.core.schemas.models import TransformComponent
from serapeum.core.models import PydanticProgramMode


@dataclass
class _Settings:
    """Settings for the Llama Index, lazily initialized."""

    _llm: Optional[LLM] = None
    _transformations: Optional[List[TransformComponent]] = None

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


    @property
    def chunk_overlap(self) -> int:
        """Get the chunk overlap."""
        if hasattr(self.node_parser, "chunk_overlap"):
            return self.node_parser.chunk_overlap
        else:
            raise ValueError("Configured node parser does not have chunk overlap.")

    @chunk_overlap.setter
    def chunk_overlap(self, chunk_overlap: int) -> None:
        """Set the chunk overlap."""
        if hasattr(self.node_parser, "chunk_overlap"):
            self.node_parser.chunk_overlap = chunk_overlap
        else:
            raise ValueError("Configured node parser does not have chunk overlap.")


# Singleton
Settings = _Settings()
