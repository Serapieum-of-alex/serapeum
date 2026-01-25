from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Generator, Generic, List, Type, Union

from pydantic import BaseModel

from serapeum.core.models import Model


class BasePydanticLLM(ABC, Generic[Model]):
    """
    A base class for LLM-powered function that return a pydantic model.
    """

    @property
    @abstractmethod
    def output_cls(self) -> Type[Model]:
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Union[Model, List[Model]]:
        pass

    async def acall(self, *args: Any, **kwargs: Any) -> Union[Model, List[Model]]:
        return self(*args, **kwargs)

    def stream_call(
        self, *args: Any, **kwargs: Any
    ) -> Generator[
        Union[Model, List[Model], "BaseModel", List["BaseModel"]], None, None
    ]:
        raise NotImplementedError("stream_call is not supported by default.")

    async def astream_call(
        self, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[
        Union[Model, List[Model], "BaseModel", List["BaseModel"]], None
    ]:
        raise NotImplementedError("astream_call is not supported by default.")
