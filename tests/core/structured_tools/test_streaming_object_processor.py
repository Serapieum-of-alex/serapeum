"""Comprehensive unit tests for StreamingObjectProcessor.

This suite focuses on method-level behavior, covering all important scenarios
for each method of StreamingObjectProcessor. Each test includes a docstring
explaining the inputs, expected results, and the aspects being validated.
"""

from __future__ import annotations

from typing import Any, List, Optional

import pytest
from pydantic import BaseModel, Field

from serapeum.core.base.llms.models import ChatResponse, Message, MessageRole
from serapeum.core.structured_tools.utils import StreamingObjectProcessor, FlexibleModel


class Person(BaseModel):
    """Simple schema for testing.

    - name: required string
    - age: optional integer
    - hobbies: list of strings with a default empty list
    """

    name: str
    age: Optional[int] = None
    hobbies: List[str] = Field(default_factory=list)


class MockLLMReturns:
    """Mock LLM that returns specified tool calls with .tool_kwargs.

    The get_tool_calls_from_response returns a list of lightweight objects
    that carry a 'tool_kwargs' attribute, mimicking the expected interface.
    """

    def __init__(self, calls: Optional[list[dict[str, Any]]] = None) -> None:
        self._calls = calls

    def get_tool_calls_from_response(self, *args: Any, **kwargs: Any):  # noqa: D401
        # Return a list of objects each having a 'tool_kwargs' attribute or None
        if not self._calls:
            return []
        return [type("TC", (), {"tool_kwargs": c}) for c in self._calls]


class TestExtractArgs:
    """Tests for _extract_args method."""

    def test_extract_args_no_tool_calls(self) -> None:
        """When additional_kwargs has no tool_calls, returns [message.content].

        Inputs:
            - ChatResponse with Message content set to a JSON string and no tool_calls.
        Expected:
            - A single-element list containing the content string is returned.
        Checks:
            - Verifies the returned args equal [content_string].
        """
        content = '{"name": "John", "age": 33}'
        resp = ChatResponse(
            message=Message(role=MessageRole.ASSISTANT, content=content)
        )
        p = StreamingObjectProcessor(Person)
        args = p._extract_args(resp)
        assert args == [content]

    def test_extract_args_tool_calls_present_but_no_llm_raises(self) -> None:
        """If tool_calls exist but no LLM is provided, a ValueError is raised.

        Inputs:
            - ChatResponse with additional_kwargs.tool_calls present (a list).
            - Processor instantiated without llm.
        Expected:
            - ValueError is raised.
        Checks:
            - Ensures correct error behavior for missing LLM dependency.
        """
        resp = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                content="",
                additional_kwargs={"tool_calls": [{}]},
            )
        )
        p = StreamingObjectProcessor(Person)
        with pytest.raises(ValueError):
            _ = p._extract_args(resp)

    def test_extract_args_tool_calls_not_list_returns_empty_parsing_obj(self) -> None:
        """If tool_calls is not a list, returns a single default parsing object.

        Inputs:
            - ChatResponse with additional_kwargs.tool_calls as a non-list (e.g., dict).
            - Processor defined with flexible parsing class (default).
        Expected:
            - Returns a list with a single instance of the parsing class (flexible model).
        Checks:
            - Type of returned element is of the dynamically created flexible model.
        """
        resp = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                content="",
                additional_kwargs={"tool_calls": {"unexpected": True}},
            )
        )
        p = StreamingObjectProcessor(Person, llm=MockLLMReturns())
        args = p._extract_args(resp)
        # The method returns [self._parsing_cls()] in this case
        assert len(args) == 1
        assert isinstance(args[0], p._parsing_cls)

    def test_extract_args_tool_calls_list_llm_returns_items(self) -> None:
        """When tool_calls is a list and LLM returns calls, return their kwargs.

        Inputs:
            - ChatResponse with additional_kwargs.tool_calls as a list.
            - Mock LLM that returns two calls with tool_kwargs.
        Expected:
            - Returns list of tool_kwargs dicts in the same order.
        Checks:
            - Content equality of returned args.
        """
        resp = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                content="",
                additional_kwargs={"tool_calls": [{}, {}]},
            )
        )
        llm = MockLLMReturns(calls=[{"name": "A"}, {"name": "B", "age": 20}])
        p = StreamingObjectProcessor(Person, llm=llm)
        args = p._extract_args(resp)
        assert args == [{"name": "A"}, {"name": "B", "age": 20}]

    def test_extract_args_tool_calls_list_llm_returns_empty(self) -> None:
        """When tool_calls is a list but LLM returns no calls, return default obj.

        Inputs:
            - ChatResponse with additional_kwargs.tool_calls as a list.
            - Mock LLM configured to return no calls.
        Expected:
            - Returns [parsing_cls()], i.e., a single default flexible model instance.
        Checks:
            - Returned length is 1 and instance type matches flexible model.
        """
        resp = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                content="",
                additional_kwargs={"tool_calls": [{"any": 1}]},
            )
        )
        p = StreamingObjectProcessor(Person, llm=MockLLMReturns(calls=[]))
        args = p._extract_args(resp)
        assert len(args) == 1
        assert isinstance(args[0], p._parsing_cls)


class TestParseSingle:
    def test_parse_single_with_valid_mapping(self) -> None:
        """Parses a valid mapping directly into the flexible parsing model.

        Inputs:
            - Dict with fields matching the Person schema.
        Expected:
            - Returns an instance of the flexible parsing class with the data set.
        Checks:
            - Instance type and field values are correctly parsed.
        """
        p = StreamingObjectProcessor(Person)
        result = p._parse_single({"name": "John", "age": 30})
        assert result is not None
        assert result.model_dump().get("name") == "John"
        assert result.model_dump().get("age") == 30

    def test_parse_single_repairs_incomplete_json_string(self) -> None:
        """Repairs and parses an incomplete JSON string argument.

        Inputs:
            - A string with missing trailing braces: '{"name": "Jane"'.
        Expected:
            - The internal repair function adds '}' and parsing succeeds.
        Checks:
            - Returned model has the expected field values.
        """
        p = StreamingObjectProcessor(Person)
        arg = '{"name": "Jane"'
        result = p._parse_single(arg)
        assert result is not None
        assert result.model_dump().get("name") == "Jane"

    def test_parse_single_unparseable_string_returns_none(self) -> None:
        """Unparseable string input returns None after repair attempt fails.

        Inputs:
            - A non-JSON string like 'not json'.
        Expected:
            - Repair does not help and model_validate_json raises internally.
        Checks:
            - Method returns None.
        """
        p = StreamingObjectProcessor(Person)
        result = p._parse_single("not json")
        assert result is None

    def test_parse_single_none_returns_none(self) -> None:
        """None input is not a valid mapping or JSON; returns None.

        Inputs:
            - None.
        Expected:
            - model_validate on None fails, string parsing path is skipped.
        Checks:
            - Method returns None.
        """
        p = StreamingObjectProcessor(Person)
        result = p._parse_single(None)  # type: ignore[arg-type]
        assert result is None
