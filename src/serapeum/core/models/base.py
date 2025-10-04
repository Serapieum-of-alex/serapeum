from typing import Any, Dict, Self
import json
import pickle
import logging
from enum import Enum
from typing import TypeVar
from pydantic_core import CoreSchema
from pydantic.json_schema import JsonSchemaValue
from pydantic import BaseModel, GetJsonSchemaHandler, model_serializer, SerializerFunctionWrapHandler, SerializationInfo


Model = TypeVar("Model", bound=BaseModel)

__all__ = ["SerializableModel", "Model", "PydanticProgramMode"]


logger = logging.getLogger(__name__)


class SerializableModel(BaseModel):
    """Serialization and Deserialization functionality."""

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """
        Overrides Pydantic's JSON schema generation to include a class_name property
        """
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
        """
        - This is meant to be overridden by the subclass.class_name() method.
        - Automatically injects the class name into the schema and serialized data.
        """
        return "base_component"

    def json(self, **kwargs: Any) -> str:
        """alias to to_json"""
        return self.to_json(**kwargs)

    @model_serializer(mode="wrap")
    def custom_model_dump(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> Dict[str, Any]:
        data = handler(self)
        data["class_name"] = self.class_name()
        return data

    def dict(self, **kwargs: Any) -> Dict[str, Any]:
        """alias to model_dump"""
        return self.model_dump(**kwargs)

    def __getstate__(self) -> Dict[str, Any]:
        """
        - Custom pickling to remove unpickleable attributes.
        - Scan both __dict__ and __pydantic_private__ for unpickleable attributes.
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

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Custom unpickling to remove unpickleable attributes to reconstruct the class via __init__.
        """
        # Use the __dict__ and __init__ method to set state
        # so that all variables initialize
        try:
            self.__init__(**state["__dict__"])  # type: ignore
        except Exception:
            # Fall back to the default __setstate__ method
            # This may not work if the class had unpickleable attributes
            super().__setstate__(state)

    def to_dict(self, **kwargs: Any) -> Dict[str, Any]:
        data = self.dict(**kwargs)
        data["class_name"] = self.class_name()
        return data

    def to_json(self, **kwargs: Any) -> str:
        data = self.to_dict(**kwargs)
        return json.dumps(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> Self:
        """Creates an instance from a dictionary (Deserialization)."""
        data = dict(data)
        if isinstance(kwargs, dict):
            data.update(kwargs)
        data.pop("class_name", None)
        return cls(**data)

    @classmethod
    def from_json(cls, data_str: str, **kwargs: Any) -> Self:
        """Creates an instance from a JSON string (Deserialization)."""
        data = json.loads(data_str)
        return cls.from_dict(data, **kwargs)


class PydanticProgramMode(str, Enum):
    """Pydantic program mode."""

    DEFAULT = "default"
    OPENAI = "openai"
    LLM = "llm"
    FUNCTION = "function"
    GUIDANCE = "guidance"