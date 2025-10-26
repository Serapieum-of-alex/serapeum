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


class TestRepairIncompleteJson:
    def test_adds_missing_quote(self) -> None:
        """Given a JSON string ending with an odd number of quotes,
        the function should append a closing quote and any missing braces.
        Input: '{"name": "John'
        Expected: '{"name": "John"}' (closing quote and brace added).
        """
        assert _repair_incomplete_json('{"name": "John') == '{"name": "John"}'

    def test_adds_missing_brace(self) -> None:
        """Given a JSON string with unbalanced opening braces,
        the function should add the required number of closing braces.
        Input: '{"name": "John"'
        Expected: '{"name": "John"}' (one closing brace added).
        """
        assert _repair_incomplete_json('{"name": "John"') == '{"name": "John"}'

    def test_empty_string(self) -> None:
        """Given an empty string, return '{}' as a minimally valid JSON object."""
        assert _repair_incomplete_json("") == "{}"

    def test_valid_json_unchanged(self) -> None:
        """A valid JSON string must be returned unchanged.
        Input: '{"name": "John", "age": 30}'
        Expected: the same string.
        """
        s = '{"name": "John", "age": 30}'
        assert _repair_incomplete_json(s) == s

    def test_multiple_unbalanced_braces(self) -> None:
        """If the input has more opening than closing braces, add enough braces
        to balance it.
        Input: '{{"a": 1'
        Expected: '{{"a": 1}}' (two closing braces added).
        """
        assert _repair_incomplete_json('{{"a": 1') == '{{"a": 1}}'

    def test_repair_incomplete_json(self) -> None:
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



class TestFlexibleModel:
    def test_allows_extra_fields(self) -> None:
        """A subclass of FlexibleModel should allow extra fields without raising.
        Input: Instantiate a subclass with an extra field.
        Expected: Instance created and attribute present.
        """
        class Sub(FlexibleModel):
            a: int

        s = Sub(a=1, extra="ok")
        assert s.a == 1
        assert getattr(s, "extra") == "ok"


class TestCreateFlexibleModel:
    def test_accepts_partial_and_extra(self) -> None:
        """The dynamically created flexible model should accept partial data and
        extra fields, defaulting missing declared fields to None.
        Input: name only, and later with an extra field.
        Expected: 'age' defaults to None and extra fields are kept.
        """
        FlexiblePerson = FlexibleModel.create(Person)
        p = FlexiblePerson(name="John")
        assert p.name == "John"
        assert p.age is None

        p2 = FlexiblePerson(name="John", extra_field="v")
        assert p2.name == "John"
        assert getattr(p2, "extra_field") == "v"

    def test_defaults_are_none(self) -> None:
        """All fields in the flexible version should be Optional[...] with default None.
        Input: Create FlexiblePerson and inspect defaults.
        Expected: Missing declared attributes are None and model accepts extras.
        """
        FlexiblePerson = FlexibleModel.create(Person)
        p = FlexiblePerson()
        assert p.name is None
        assert p.age is None
        assert p.hobbies is None
        # Extra field still allowed
        p.extra = 123
        assert getattr(p, "extra") == 123

    def test_create_flexible_model(self) -> None:
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
