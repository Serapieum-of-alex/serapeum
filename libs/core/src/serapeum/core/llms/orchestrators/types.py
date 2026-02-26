"""Base classes for LLM-powered structured tools using Pydantic models."""

from abc import ABC, abstractmethod
from typing import Any, Generic, List, Type, Union

from pydantic import BaseModel

from serapeum.core.types import Model


class BasePydanticLLM(ABC, Generic[Model]):
    """A base class for LLM-powered function that return a pydantic model."""

    @property
    @abstractmethod
    def schema(self) -> Type[Model]:
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Union[Model, List[Model]]:
        pass

    async def acall(self, *args: Any, **kwargs: Any) -> Union[Model, List[Model]]:
        return self(*args, **kwargs)
