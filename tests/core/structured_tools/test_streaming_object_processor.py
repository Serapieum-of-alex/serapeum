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


class TestParseObjects:

    def test_parse_objects_filters_invalid_and_uses_fallback(self) -> None:
        """Parses only valid args; when none valid, use fallback if provided.

        Inputs:
            - args with one valid dict, one invalid string, and one None.
            - fallback list with a Person("Prev", 40).
        Expected:
            - Only the valid dict is parsed when present.
            - When args are all invalid, returns the fallback list.
        Checks:
            - Correct list length and content for both cases.
        """
        p = StreamingObjectProcessor(Person)
        args = [{"name": "Ok", "age": 1}, "not json", None]
        parsed = p._parse_objects(args, fallback=None)
        assert len(parsed) == 1 and parsed[0].model_dump().get("name") == "Ok"

        parsed2 = p._parse_objects(["bad", None], fallback=[Person(name="Prev", age=40)])
        assert len(parsed2) == 1 and isinstance(parsed2[0], Person)
        assert parsed2[0].name == "Prev" and parsed2[0].age == 40

    def test_parse_objects_empty_and_no_fallback_returns_empty_instance(self) -> None:
        """When nothing parses and no fallback, returns a single default instance.

        Inputs:
            - args empty list, fallback None.
        Expected:
            - Returns a list with one element: the flexible parsing class default.
        Checks:
            - Length is 1 and type is flexible model.
        """
        p = StreamingObjectProcessor(Person)
        parsed = p._parse_objects([], fallback=None)
        assert len(parsed) == 1 and isinstance(parsed[0], p._parsing_cls)


class TestSelectBest:

    def test_select_best_prefers_more_complete_objects(self) -> None:
        """Selects the object set with greater number of valid fields.

        Inputs:
            - cur_objects: one Person with only name.
            - new_objects: one Person with name and age (more fields).
        Expected:
            - new_objects is selected.
        Checks:
            - Equality by identity with the chosen list.
        """
        p = StreamingObjectProcessor(Person)
        cur = [Person(name="A")]
        new = [Person(name="A", age=10)]
        chosen = p._select_best(new_objects=new, cur_objects=cur)
        assert chosen is new

    def test_select_best_equal_fields_prefers_new(self) -> None:
        """When equal number of valid fields, prefers new_objects (>= condition).

        Inputs:
            - cur_objects and new_objects both having one field set.
        Expected:
            - Returns new_objects as per implementation (>= uses new on tie).
        Checks:
            - Identity equals new_objects.
        """
        p = StreamingObjectProcessor(Person)
        cur = [Person(name="A")]
        new = [Person(name="B")]
        chosen = p._select_best(new_objects=new, cur_objects=cur)
        assert chosen is new

    def test_select_best_when_no_current_returns_new(self) -> None:
        """If cur_objects is None, returns new_objects.

        Inputs:
            - cur_objects: None
            - new_objects: list with one Person
        Expected:
            - new_objects returned unchanged.
        Checks:
            - Identity equals new_objects.
        """
        p = StreamingObjectProcessor(Person)
        new = [Person(name="A")]
        chosen = p._select_best(new_objects=new, cur_objects=None)
        assert chosen is new


class TestFinalize:

    def test_finalize_converts_flexible_to_strict_when_valid(self) -> None:
        """In flexible mode, converts flexible objects to the strict output schema.

        Inputs:
            - A flexible instance with valid fields for Person.
        Expected:
            - Conversion to Person succeeds and returned list contains Person.
        Checks:
            - Instance type is Person and values preserved.
        """
        p = StreamingObjectProcessor(Person, flexible_mode=True)
        FlexiblePerson = FlexibleModel.create(Person)
        flex = FlexiblePerson(name="John", age=44)
        finalized = p._finalize([flex])
        assert isinstance(finalized[0], Person)
        assert finalized[0].name == "John" and finalized[0].age == 44

    def test_finalize_keeps_flexible_when_conversion_fails(self) -> None:
        """If conversion to strict schema fails, keeps the flexible instance.

        Inputs:
            - A flexible instance missing required 'name'.
        Expected:
            - ValidationError inside _finalize causes original flexible to be kept.
        Checks:
            - Returned instance is still the flexible model type.
        """
        p = StreamingObjectProcessor(Person, flexible_mode=True)
        FlexiblePerson = FlexibleModel.create(Person)
        flex = FlexiblePerson(age=12)  # missing required name for strict Person
        finalized = p._finalize([flex])
        assert isinstance(finalized[0], FlexiblePerson)

    def test_finalize_noop_when_not_flexible_mode(self) -> None:
        """When flexible_mode is False, _finalize returns objects unchanged.

        Inputs:
            - Processor configured with flexible_mode=False and a Person list.
        Expected:
            - Returned list is identical to input (by identity).
        Checks:
            - Identity equality.
        """
        p = StreamingObjectProcessor(Person, flexible_mode=False)
        objs = [Person(name="X")]
        finalized = p._finalize(objs)
        assert finalized is objs


class TestFormatOutput:
    def test_format_output_allow_parallel_true_returns_list(self) -> None:
        """When allow_parallel is True, returns the objects list as-is.

        Inputs:
            - Processor with allow_parallel_tool_calls=True and list of two Persons.
        Expected:
            - The same list object is returned.
        Checks:
            - Identity equality.
        """
        p = StreamingObjectProcessor(Person, allow_parallel_tool_calls=True)
        objs = [Person(name="A"), Person(name="B")]
        out = p._format_output(objs)
        assert out is objs

    def test_format_output_allow_parallel_false_returns_first_and_warns(self, caplog: Any) -> None:
        """When allow_parallel is False, returns the first object and logs a warning.

        Inputs:
            - Processor with default allow_parallel=False and list of two Persons.
        Expected:
            - Returns only the first Person, and a warning is recorded in logs.
        Checks:
            - Returned instance equals first; caplog captured a warning message.
        """
        p = StreamingObjectProcessor(Person)
        objs = [Person(name="A"), Person(name="B")]
        with caplog.at_level("WARNING"):
            out = p._format_output(objs)
        assert isinstance(out, Person) and out.name == "A"
        # Ensure a warning about multiple outputs was logged
        assert any("Multiple outputs found" in rec.message for rec in caplog.records)

    def test_format_output_empty_list_returns_empty_list(self) -> None:
        """If objects list is empty and not allowing parallel, returns empty list.

        Inputs:
            - Empty objects list.
        Expected:
            - Returns the same empty list (falsy), not raising errors.
        Checks:
            - Equality to the input empty list.
        """
        p = StreamingObjectProcessor(Person)
        out = p._format_output([])
        assert out == []


