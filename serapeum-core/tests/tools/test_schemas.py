from pydantic import BaseModel

from serapeum.core.tools.models import MinimalToolSchema
from serapeum.core.utils.schemas import Schema


class TestSchema:
    """Test suite for Schema."""

    def test_default_schema_filtered(self):
        """Test that default Pydantic schema is filtered to allowed keys.

        Inputs:
          - ToolMetadata with tool_schema=MinimalToolSchema.
        Expected:
          - The schema dict only contains filtered keys and has an `input` string property.
        Checks:
          - Only expected keys exist; `input` under properties is present and string-typed.
        """
        schema = Schema(full_schema=MinimalToolSchema.model_json_schema())
        params = schema.resolve_references()
        for k in params.keys():
            assert k in {"type", "properties", "required", "definitions", "$defs"}
        assert params["properties"]["input"]["type"] == "string"

    def test_nested_schema_includes_defs(self):
        """Test that nested Pydantic models preserve $defs/definitions when present.

        Inputs:
          - A custom schema with a nested sub-model to generate $defs/definitions.
        Expected:
          - get_parameters_dict returns filtered schema that still includes "$defs" or "definitions".
        Checks:
          - Either "$defs" or "definitions" key exists in the returned dict.
        """

        class SubModel(BaseModel):
            x: int

        class MainModel(BaseModel):
            sub: SubModel

        schema = Schema(full_schema=MainModel.model_json_schema())
        params = schema.resolve_references()
        assert params == {
            "$defs": {
                "SubModel": {
                    "properties": {"x": {"title": "X", "type": "integer"}},
                    "required": ["x"],
                    "title": "SubModel",
                    "type": "object",
                }
            },
            "properties": {"sub": {"$ref": "#/$defs/SubModel"}},
            "required": ["sub"],
            "type": "object",
        }

        params = schema.resolve_references(inline=True)
        assert params == {
            "properties": {
                "sub": {
                    "properties": {"x": {"title": "X", "type": "integer"}},
                    "required": ["x"],
                    "title": "SubModel",
                    "type": "object",
                }
            },
            "required": ["sub"],
            "type": "object",
        }
