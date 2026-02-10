"""Pydantic base classes and serialization helpers used across Serapeum."""

from __future__ import annotations

import builtins
import json
import logging
import pickle  # nosec B403
from enum import Enum
from typing import Any, Self, TypeVar

from pydantic import (
    BaseModel,
    GetJsonSchemaHandler,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    model_serializer,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

Model = TypeVar("Model", bound=BaseModel)

__all__ = ["SerializableModel", "Model", "StructuredLLMMode"]


logger = logging.getLogger(__name__)


class SerializableModel(BaseModel):
    """Serialization and deserialization helpers for Pydantic models."""

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Override JSON schema generation to include a ``class_name`` property."""
        json_schema = handler(core_schema)
        json_schema = handler.resolve_ref_schema(json_schema)

        # inject class name to help with serde
        if "properties" in json_schema:
            json_schema["properties"]["class_name"] = {
                "title": "Class Name",
                "type": "string",
                "default": cls.class_name(),
            }
        return json_schema

    @classmethod
    def class_name(cls) -> str:
        """Return a stable class name identifier for serialization.

        Subclasses should override to provide a meaningful identifier. The
        value is injected into the JSON schema and serialized payloads to aid
        with round‑tripping.
        """
        return "base_component"

    def json(self, **kwargs: Any) -> str:
        """Alias to :meth:`to_json`."""
        return self.to_json(**kwargs)

    @model_serializer(mode="wrap")
    def custom_model_dump(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> builtins.dict[str, Any]:
        data = handler(self)
        data["class_name"] = self.class_name()
        return data

    def dict(self, **kwargs: Any) -> builtins.dict[str, Any]:
        """Alias to :meth:`model_dump`."""
        return self.model_dump(**kwargs)

    def __getstate__(self) -> builtins.dict[str, Any]:
        """Return a picklable state, pruning unpickleable attributes.

        Scans both ``__dict__`` and ``__pydantic_private__`` for values that
        cannot be pickled and removes them from the state.
        """
        state = super().__getstate__()

        # remove attributes that are not pickleable -- kind of dangerous
        keys_to_remove = []
        for key, val in state["__dict__"].items():
            try:
                pickle.dumps(val)
            except Exception:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            logging.warning(f"Removing unpickleable attribute {key}")
            del state["__dict__"][key]

        # remove private attributes if they aren't pickleable -- kind of dangerous
        keys_to_remove = []
        private_attrs = state.get("__pydantic_private__", None)
        if private_attrs:
            for key, val in state["__pydantic_private__"].items():
                try:
                    pickle.dumps(val)
                except Exception:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                logging.warning(f"Removing unpickleable private attribute {key}")
                del state["__pydantic_private__"][key]

        return state

    def __setstate__(self, state: builtins.dict[str, Any]) -> None:
        """Reconstruct instance from pickled state safely."""
        # Use the __dict__ and __init__ method to set state
        # so that all variables initialize
        try:
            self.__init__(**state["__dict__"])  # type: ignore
        except Exception:
            # Fall back to the default __setstate__ method
            # This may not work if the class had unpickleable attributes
            super().__setstate__(state)

    def to_dict(self, **kwargs: Any) -> builtins.dict[str, Any]:
        data = self.dict(**kwargs)
        data["class_name"] = self.class_name()
        return data

    def to_json(self, **kwargs: Any) -> str:
        data = self.to_dict(**kwargs)
        return json.dumps(data)

    @classmethod
    def from_dict(cls, data: builtins.dict[str, Any], **kwargs: Any) -> Self:
        """Create an instance from a dictionary (deserialization)."""
        data = dict(data)
        if isinstance(kwargs, dict):
            data.update(kwargs)
        data.pop("class_name", None)
        return cls(**data)

    @classmethod
    def from_json(cls, data_str: str, **kwargs: Any) -> Self:
        """Create an instance from a JSON string (deserialization)."""
        data = json.loads(data_str)
        return cls.from_dict(data, **kwargs)


class StructuredLLMMode(str, Enum):
    """Pydantic program mode."""

    DEFAULT = "default"
    OPENAI = "openai"
    LLM = "llm"
    FUNCTION = "function"
    GUIDANCE = "guidance"
