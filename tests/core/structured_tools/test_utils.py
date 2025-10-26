"""Test program utils."""
import pytest
from typing import List, Optional
from pydantic import BaseModel, Field
from serapeum.core.base.llms.models import Message, ChatResponse, MessageRole
from serapeum.core.structured_tools.utils import (
    _repair_incomplete_json,
    StreamingObjectProcessor,
    num_valid_fields,
    FlexibleModel
)


class Person(BaseModel):
    name: str
    age: Optional[int] = None
    hobbies: List[str] = Field(default_factory=list)


def test_repair_incomplete_json() -> None:
    """Test JSON repair function."""
    # Test adding missing quotes
    assert _repair_incomplete_json('{"name": "John') == '{"name": "John"}'

    # Test adding missing braces
    assert _repair_incomplete_json('{"name": "John"') == '{"name": "John"}'

    # Test empty string
    assert _repair_incomplete_json("") == "{}"

    # Test already valid JSON
    valid_json = '{"name": "John", "age": 30}'
    assert _repair_incomplete_json(valid_json) == valid_json


def test_process_streaming_objects() -> None:
    """Test processing streaming objects."""
    # Test processing complete object
    response = ChatResponse(
        message=Message(
            role=MessageRole.ASSISTANT,
            content='{"name": "John", "age": 30}',
        )
    )

    processor = StreamingObjectProcessor(output_cls=Person)
    result = processor.process(response)

    assert isinstance(result, Person)
    assert result.name == "John"
    assert result.age == 30

    # Test processing incomplete object
    incomplete_response = ChatResponse(
        message=Message(
            role=MessageRole.ASSISTANT,
            content='{"name": "John", "age":',
        )
    )

    # Should return empty object when can't parse
    processor = StreamingObjectProcessor(output_cls=Person)
    result = processor.process(incomplete_response)

    assert result.name is None  # Default value

    # Test with previous state
    prev_obj = Person(name="John", age=25)
    processor = StreamingObjectProcessor(output_cls=Person)
    result = processor.process(incomplete_response, [prev_obj])

    assert isinstance(result, Person)
    assert result.name == "John"
    assert result.age == 25  # Keeps previous state

    # Test with tool calls
    tool_call_response = ChatResponse(
        message=Message(
            role=MessageRole.ASSISTANT,
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "function": {
                            "name": "create_person",
                            "arguments": '{"name": "Jane", "age": 28}',
                        }
                    }
                ]
            },
        )
    )

    # Mock LLM for tool calls
    class MockLLM:
        def get_tool_calls_from_response(self, *args, **kwargs):
            return [
                type(
                    "ToolCallArguments",
                    (),
                    {"tool_kwargs": {"name": "Jane", "age": 28}},
                )
            ]

    processor = StreamingObjectProcessor(
        output_cls=Person,
        llm=MockLLM(), # type: ignore
    )
    result = processor.process(tool_call_response)

    assert isinstance(result, Person)
    assert result.name == "Jane"
    assert result.age == 28


class TestNumValidFields:
    def test_none_and_scalars(self) -> None:
        """None should count as 0; any non-None scalar should count as 1.
        Input: None and scalar strings.
        Expected: 0 for None, 1 for non-None.
        """
        assert num_valid_fields(None) == 0  # type: ignore[arg-type]
        assert num_valid_fields("x") == 1  # type: ignore[arg-type]

    def test_simple_model(self) -> None:
        """A model with only one set field should be counted as 1.
        Input: Person(name='John').
        Expected: 1.
        """
        p = Person(name="John")
        assert num_valid_fields(p) == 1

    def test_lists_and_dicts(self) -> None:
        """Lists and dicts should be counted recursively across elements/values.
        Input: list[Person], dict[str, Person].
        Expected: sum of non-None fields across items.
        """
        p1 = Person(name="John", age=30)
        p2 = Person(name="Jane")
        assert num_valid_fields([p1, p2]) == 3
        assert num_valid_fields({"a": p1, "b": p2}) == 3

    def test_nested_models(self) -> None:
        """Nested Pydantic models should be traversed to count non-None fields.
        Input: Family with parent and children Persons.
        Expected: count equals sum of all non-None nesting fields.
        """
        class Family(BaseModel):
            parent: Person
            children: List[Person] = Field(default_factory=list)

        fam = Family(parent=Person(name="P", age=40), children=[Person(name="C")])
        # parent has 2, child has 1
        assert num_valid_fields(fam) == 3

    def test_num_valid_fields(self) -> None:
        """Test counting valid fields."""
        # Test simple object
        person = Person(name="John", age=None, hobbies=[])
        assert num_valid_fields(person) == 1  # Only name is non-None

        # Test with more fields
        person = Person(name="John", age=30, hobbies=["reading"])
        assert num_valid_fields(person) == 3  # All fields are non-None

        # Test list of objects
        people = [
            Person(name="John", age=30),
            Person(name="Jane", hobbies=["reading"]),
        ]
        assert num_valid_fields(people) == 4  # 2 names + 1 age + 1 hobby list

        # Test nested object
        class Family(BaseModel):
            parent: Person
            children: List[Person] = []

        family = Family(
            parent=Person(name="John", age=40),
            children=[Person(name="Jane", age=10)],
        )
        assert num_valid_fields(family) == 4  # parent's name & age + child's name & age


def test_create_flexible_model() -> None:
    """Test creating flexible model."""
    FlexiblePerson = FlexibleModel.create(Person)

    # Should accept partial data
    flexible_person = FlexiblePerson(name="John")
    assert flexible_person.name == "John"
    assert flexible_person.age is None

    # Should accept extra fields
    flexible_person = FlexiblePerson(
        name="John", extra_field="value", another_field=123
    )
    assert flexible_person.name == "John"
    assert hasattr(flexible_person, "extra_field")
    assert flexible_person.extra_field == "value"

    # Original model should still be strict
    with pytest.raises(ValueError):
        Person(name=None)  # type: ignore
