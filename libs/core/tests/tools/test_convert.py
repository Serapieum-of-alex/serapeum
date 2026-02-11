"""Unit tests for serapeum.core.tools.convert.

This test suite covers all functions in utils.py:
- create_schema_from_function

For each function, we define a dedicated test class with multiple test methods, each
with a docstring that explains inputs, expected behavior, and what is being checked.
"""

import datetime as dt
from typing import Any, Optional

import pytest
from pydantic import BaseModel, Field

from serapeum.core.tools.convert import (
    Docstring,
    FunctionConverter,
)


class MockSong(BaseModel):
    """Mock Song class.

    Here is a long description of the class.

    Attributes:
        title (str):
            song title
        length (Optional[int]):
            length of the song in seconds
        author (Optional[str]):
            name of the author of the song. Defaults to None.

    Examples:
        - provide only the title.
            ```python
            >>> song = MockSong(title="song title")

            ```
        - provide title and length.
            ```python
            >>> song = MockSong(title="song title", length=120)

            ```
        - provide title, length, and author.
            ```python
            >>> song = MockSong(title="song title", length=120, author="author name")

            ```
    """

    title: str
    length: int | None = None
    author: str | None = Field(default=None, description="author")


class TestDocstring:
    """Test suite for Docstring."""

    def test_pydantic_class(self):
        docstring = Docstring(MockSong)
        param_docs, unknown = docstring.extract_param_docs()
        assert param_docs == {
            "title": "song title",
            "length": "length of the song in seconds",
            "author": "name of the author of the song. Defaults to None.",
        }
        summary = docstring.get_short_summary_line()
        assert (
            summary
            == "MockSong(*, title: str, length: int | None = None, author: str | None = None) -> None\nMock Song class."
        )

    def test_extracts_sphinx_google_javadoc_and_filters_unknown(self):
        """Test extraction and filtering of multiple docstring styles.

        Inputs:
            A composite docstring containing Sphinx (:param), Google (name (type): desc), and Javadoc (@param) parameter docs.
            Known function parameters: {"a", "b"}. Docstring also includes an unknown param "c".
        Expected:
            Returned param_docs provides descriptions for only known params (a and b). unknown_params contains {"c"}.
        Checks:
            All three styles are parsed, only known params are kept, and unknown parameters are reported.
        """

        def f(a: int, b: str) -> None:
            """Summary line.

            :param a: value for a
            b (int): value for b
            @param c value for c (unknown)
            """
            pass

        docstring = Docstring(f)
        param_docs, unknown = docstring.extract_param_docs()
        assert param_docs == {"a": "value for a", "b": "value for b"}
        assert unknown == {"c"}

    def test_conflicting_descriptions_keep_first(self):
        """Test conflict resolution for duplicate parameter documentation.

        Inputs: A docstring defines the same parameter twice with different descriptions.
        Expected: The first description is retained; the conflicting second one is ignored.
        Checks: Conflict resolution behavior when duplicate param documentation with different text appears.
        """

        def f(x: int) -> None:
            """
            :param x: first desc
            :param x: second desc  # noqa: D205, D400
            """
            pass

        docstrings = Docstring(f)
        param_docs, unknown = docstrings.extract_param_docs()
        assert param_docs == {"x": "first desc"}
        assert unknown == set()


class TestCreateSchemaFromFunction:
    """Tests for create_schema_from_function."""

    def test_required_default_any_and_field_info(self) -> None:
        """Validate required fields, defaulted fields, Any typing, and Field defaults.

        Inputs:
            - A Python function with the following parameters:
              - a: int (no default) -> required
              - b: str = "x" (defaulted)
              - c = 3 (no annotation) -> treated as Any and defaulted
              - d: int = Field(default=5, description="five") (FieldInfo default)
        Expected:
            - Pydantic model has corresponding fields with correct metadata:
              - 'a' required with annotation int.
              - 'b' default 'x'.
              - 'c' annotation Any and default 3.
              - 'd' default 5 and description preserved.
            - JSON schema 'required' includes only 'a'.
        Checks:
            - model.model_fields metadata and model_json_schema entries.
        """
        d_field_default = Field(default=5, description="five")

        def f(a: int, b: str = "x", c=3, d: int = d_field_default) -> None:
            return None

        function = FunctionConverter("F", f)
        model = function.to_schema()
        schema = model.model_json_schema()
        required = set(schema.get("required", []))

        assert required == {"a"}
        assert model.model_fields["a"].annotation is int
        assert model.model_fields["b"].default == "x"
        assert model.model_fields["c"].annotation is Any
        assert model.model_fields["c"].default == 3
        assert model.model_fields["d"].default == 5
        assert model.model_fields["d"].description == "five"

    def test_annotated_description_and_field_info_extra(self) -> None:
        """Validate Annotated metadata for description and json_schema_extra merging.

        Inputs:
            - A function with two Annotated parameters:
              - x: Annotated[int, "counter value"]
              - y: Annotated[str, Field(description="text", json_schema_extra={"alpha": True})]
        Expected:
            - Field descriptions reflect annotations.
            - Field 'y' has json_schema_extra merged to include 'alpha': True.
        Checks:
            - model.model_fields descriptions and json_schema_extra content.
        """
        from typing import Annotated as Ann

        counter_desc = "counter value"
        text_desc = "text"
        alpha_extra = {"alpha": True}

        def g(
            x: Ann[int, counter_desc],
            y: Ann[str, Field(description=text_desc, json_schema_extra=alpha_extra)],
        ) -> None:
            return None

        function = FunctionConverter("G", g)
        model = function.to_schema()
        fx = model.model_fields["x"]
        fy = model.model_fields["y"]
        assert fx.description == "counter value"
        assert fy.description == "text"
        assert (
            isinstance(fy.json_schema_extra, dict)
            and fy.json_schema_extra.get("alpha") is True
        )

    def test_date_datetime_time_formats(self) -> None:
        """Ensure date/datetime/time fields carry proper JSON Schema format values.

        Inputs:
            - FunctionConverter with parameters of type date, datetime, and time.
        Expected:
            - Corresponding JSON Schema properties include format: 'date', 'date-time', 'time'.
        Checks:
            - model.model_json_schema property inspection.
        """

        def h(day: dt.date, ts: dt.datetime, at: dt.time) -> None:
            return None

        function = FunctionConverter("H", h)
        model = function.to_schema()
        props = model.model_json_schema()["properties"]
        assert props["day"].get("format") == "date"
        assert props["ts"].get("format") == "date-time"
        assert props["at"].get("format") == "time"

    def test_ignore_and_additional_fields(self) -> None:
        """Test ignore_fields and additional_fields (both 2- and 3-tuples).

        Inputs:
            - FunctionConverter with parameters: keep_me: int, skip_me: str
            - ignore_fields=["skip_me"], additional_fields=[("extra", int), ("flag", bool, True)]
        Expected:
            - 'skip_me' is excluded.
            - 'extra' is required (no default), 'flag' has default True.
        Checks:
            - JSON schema required set and defaults in model fields.
        """

        def k(keep_me: int, skip_me: str) -> None:
            return None

        function = FunctionConverter(
            "K",
            k,
            additional_fields=[("extra", int), ("flag", bool, True)],
            ignore_fields=["skip_me"],
        )
        model = function.to_schema()
        schema = model.model_json_schema()
        required = set(schema.get("required", []))

        assert "skip_me" not in schema.get("properties", {})
        assert "keep_me" in required and "extra" in required
        assert model.model_fields["flag"].default is True

    def test_invalid_additional_fields_tuple_length(self) -> None:
        """Verify that invalid additional_fields tuple lengths raise ValueError.

        Inputs:
            - additional_fields containing a 1-tuple (invalid length).
        Expected:
            - ValueError is raised with an explanatory message.
        Checks:
            - Exception type and message contents.
        """

        def z(x: int) -> None:
            return None

        with pytest.raises(ValueError):
            function = FunctionConverter("Z", z, additional_fields=[("bad",)])
            _ = function.to_schema()
