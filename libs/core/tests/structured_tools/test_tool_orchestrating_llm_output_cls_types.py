"""Comprehensive tests for ToolOrchestratingLLM with different output_cls types.

This test module covers all possible ways to pass output_cls to ToolOrchestratingLLM:
- Pydantic BaseModel classes
- Regular functions (sync and async)
- Lambda functions
- Callable classes
- Dataclasses wrapped in functions
- Edge cases and error conditions

Test organization:
- Unit tests: Test _create_tool() method with mocked dependencies
- Integration tests: Test with MockLLM
- E2E tests: Test with real Ollama server
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from serapeum.core.base.llms.types import Message, Metadata
from serapeum.core.chat.models import AgentChatResponse
from serapeum.core.structured_tools import ToolOrchestratingLLM
from serapeum.core.tools import ToolOutput
from serapeum.core.tools.callable_tool import CallableTool
from serapeum.core.tools.models import BaseTool

# ============================================================================
# Test Models and Functions
# ============================================================================


class SimpleOutput(BaseModel):
    """Simple Pydantic model for testing."""

    value: str
    count: int = 0


class ComplexOutput(BaseModel):
    """Complex nested Pydantic model for testing."""

    name: str
    data: Dict[str, Any]
    items: List[str] = Field(default_factory=list)


def simple_function(text: str) -> dict:
    """Simple function that returns a dict.

    Args:
        text: Input text

    Returns:
        Dictionary with processed text
    """
    return {"result": text.upper(), "length": len(text)}


def complex_function(a: int, b: int, operation: str = "add") -> dict:
    """Function with multiple parameters and default values.

    Args:
        a: First number
        b: Second number
        operation: Operation type (add, multiply, subtract)

    Returns:
        Dictionary with operation result
    """
    if operation == "add":
        result = a + b
    elif operation == "multiply":
        result = a * b
    elif operation == "subtract":
        result = a - b
    else:
        result = 0

    return {"operation": operation, "result": result, "inputs": [a, b]}


async def async_function(value: int) -> dict:
    """Async function for testing.

    Args:
        value: Input value

    Returns:
        Dictionary with doubled value
    """
    await asyncio.sleep(0.01)  # Simulate async work
    return {"value": value, "doubled": value * 2}


@dataclass
class DataClassOutput:
    """Dataclass for testing."""

    name: str
    score: float
    tags: List[str]


def dataclass_factory(name: str, score: float, tags: List[str]) -> dict:
    """Factory function that creates dataclass and returns dict.

    Args:
        name: Name field
        score: Score field
        tags: Tags field

    Returns:
        Dictionary representation of dataclass
    """
    obj = DataClassOutput(name=name, score=score, tags=tags)
    return {"name": obj.name, "score": obj.score, "tags": obj.tags}


class CallableClass:
    """Callable class for testing."""

    def __call__(self, message: str) -> dict:
        """Process message when called.

        Args:
            message: Input message

        Returns:
            Dictionary with processed message
        """
        return {"message": message, "reversed": message[::-1], "upper": message.upper()}


# ============================================================================
# Mock LLM for Integration Tests
# ============================================================================


class MockLLM(MagicMock):
    """Mock LLM that returns predefined responses."""

    def __init__(self, return_value: Any = None, **kwargs):
        """Initialize MockLLM with optional return value."""
        super().__init__(**kwargs)
        self._return_value = return_value or {"result": "test"}

    @property
    def metadata(self) -> Metadata:
        """Return metadata indicating function calling support."""
        return Metadata(is_function_calling_model=True)

    def _extend_messages(self, messages: List[Message]) -> List[Message]:
        """Extend messages with system prompts."""
        return messages

    def predict_and_call(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, Message]] = None,
        chat_history: Optional[List[Message]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> AgentChatResponse:
        """Mock predict and call that returns predefined response."""
        tool_outputs = [
            ToolOutput(
                content=str(self._return_value),
                tool_name="test_tool",
                raw_input={},
                raw_output=self._return_value,
            )
        ]

        if allow_parallel_tool_calls:
            # Return multiple outputs for parallel calls
            tool_outputs.append(
                ToolOutput(
                    content=str(self._return_value),
                    tool_name="test_tool",
                    raw_input={},
                    raw_output=self._return_value,
                )
            )

        return AgentChatResponse(
            response="mock response",
            sources=tool_outputs,
        )

    async def apredict_and_call(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, Message]] = None,
        chat_history: Optional[List[Message]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> AgentChatResponse:
        """Async version of predict_and_call."""
        return self.predict_and_call(
            tools, user_msg, chat_history, verbose, allow_parallel_tool_calls, **kwargs
        )


# ============================================================================
# Unit Tests: Testing _create_tool() Method
# ============================================================================


class TestCreateToolMethod:
    """Unit tests for the _create_tool() method.

    Tests that the method correctly detects and handles different types
    of output_cls and creates appropriate CallableTool instances.
    """

    @pytest.mark.unit
    def test_create_tool_with_pydantic_model(self):
        """Test _create_tool() with a Pydantic BaseModel class.

        Expected: Should call CallableTool.from_model() and return a CallableTool.
        """
        llm = MockLLM()
        tools_llm = ToolOrchestratingLLM(
            output_cls=SimpleOutput,
            prompt="Test prompt",
            llm=llm,
        )

        tool = tools_llm._create_tool()

        assert isinstance(tool, CallableTool)
        assert tool.metadata.name == "SimpleOutput"

    @pytest.mark.unit
    def test_create_tool_with_regular_function(self):
        """Test _create_tool() with a regular Python function.

        Expected: Should call CallableTool.from_function() and return a CallableTool.
        """
        llm = MockLLM()
        tools_llm = ToolOrchestratingLLM(
            output_cls=simple_function,
            prompt="Test prompt",
            llm=llm,
        )

        tool = tools_llm._create_tool()

        assert isinstance(tool, CallableTool)
        assert tool.metadata.name == "simple_function"

    @pytest.mark.unit
    def test_create_tool_with_async_function(self):
        """Test _create_tool() with an async function.

        Expected: Should call CallableTool.from_function() and handle async functions.
        """
        llm = MockLLM()
        tools_llm = ToolOrchestratingLLM(
            output_cls=async_function,
            prompt="Test prompt",
            llm=llm,
        )

        tool = tools_llm._create_tool()

        assert isinstance(tool, CallableTool)
        assert tool.metadata.name == "async_function"

    @pytest.mark.unit
    def test_create_tool_with_lambda(self):
        """Test _create_tool() with a lambda function.

        Expected: Should call CallableTool.from_function() for lambda.
        """
        lambda_func = lambda x: {"value": x * 2}  # noqa: E731

        llm = MockLLM()
        tools_llm = ToolOrchestratingLLM(
            output_cls=lambda_func,
            prompt="Test prompt",
            llm=llm,
        )

        tool = tools_llm._create_tool()

        assert isinstance(tool, CallableTool)
        assert tool.metadata.name == "<lambda>"

    @pytest.mark.unit
    @pytest.mark.xfail(
        reason="CallableTool.from_function() doesn't support callable instances yet"
    )
    def test_create_tool_with_callable_class(self):
        """Test _create_tool() with a callable class instance.

        Expected: Currently fails because CallableTool.from_function() doesn't support
        callable class instances (only functions). This is a known limitation.
        """
        callable_obj = CallableClass()

        llm = MockLLM()
        tools_llm = ToolOrchestratingLLM(
            output_cls=callable_obj,
            prompt="Test prompt",
            llm=llm,
        )

        tool = tools_llm._create_tool()

        assert isinstance(tool, CallableTool)

    @pytest.mark.unit
    def test_create_tool_with_invalid_type_raises_error(self):
        """Test _create_tool() with invalid type (not model or callable).

        Expected: Should raise TypeError with descriptive message.
        """
        llm = MockLLM()
        tools_llm = ToolOrchestratingLLM(
            output_cls="invalid_string",  # type: ignore
            prompt="Test prompt",
            llm=llm,
        )

        with pytest.raises(TypeError) as exc_info:
            tools_llm._create_tool()

        assert (
            "must be either a Pydantic BaseModel subclass or a callable function"
            in str(exc_info.value)
        )

    @pytest.mark.unit
    def test_create_tool_with_none_raises_error(self):
        """Test _create_tool() with None.

        Expected: Should raise TypeError.
        """
        llm = MockLLM()
        tools_llm = ToolOrchestratingLLM(
            output_cls=None,  # type: ignore
            prompt="Test prompt",
            llm=llm,
        )

        with pytest.raises(TypeError):
            tools_llm._create_tool()

    @pytest.mark.unit
    def test_create_tool_with_integer_raises_error(self):
        """Test _create_tool() with integer.

        Expected: Should raise TypeError.
        """
        llm = MockLLM()
        tools_llm = ToolOrchestratingLLM(
            output_cls=42,  # type: ignore
            prompt="Test prompt",
            llm=llm,
        )

        with pytest.raises(TypeError):
            tools_llm._create_tool()


# ============================================================================
# Integration Tests: Pydantic Models with Mock LLM
# ============================================================================


class TestPydanticModelsIntegration:
    """Integration tests using Pydantic models with MockLLM.

    Tests the complete flow from initialization to execution with
    Pydantic models as output_cls.
    """

    @pytest.mark.integration
    def test_simple_pydantic_model_sync(self):
        """Test ToolOrchestratingLLM with simple Pydantic model (sync).

        Expected: Should successfully execute and return SimpleOutput instance.
        """
        mock_output = SimpleOutput(value="test", count=5)
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=SimpleOutput,
            prompt="Generate output for {input}",
            llm=llm,
        )

        result = tools_llm(input="test")

        assert isinstance(result, SimpleOutput)
        assert result.value == "test"
        assert result.count == 5

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_simple_pydantic_model_async(self):
        """Test ToolOrchestratingLLM with simple Pydantic model (async).

        Expected: Should successfully execute asynchronously and return SimpleOutput.
        """
        mock_output = SimpleOutput(value="async_test", count=10)
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=SimpleOutput,
            prompt="Generate output for {input}",
            llm=llm,
        )

        result = await tools_llm.acall(input="test")

        assert isinstance(result, SimpleOutput)
        assert result.value == "async_test"
        assert result.count == 10

    @pytest.mark.integration
    def test_complex_pydantic_model(self):
        """Test ToolOrchestratingLLM with complex nested Pydantic model.

        Expected: Should handle nested models correctly.
        """
        mock_output = ComplexOutput(
            name="test_complex",
            data={"key": "value", "number": 42},
            items=["item1", "item2", "item3"],
        )
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=ComplexOutput,
            prompt="Generate complex output",
            llm=llm,
        )

        result = tools_llm()

        assert isinstance(result, ComplexOutput)
        assert result.name == "test_complex"
        assert result.data["key"] == "value"
        assert len(result.items) == 3

    @pytest.mark.integration
    def test_pydantic_model_with_parallel_calls(self):
        """Test Pydantic model with parallel tool calls enabled.

        Expected: Should return a list of SimpleOutput instances.
        """
        mock_output = SimpleOutput(value="parallel", count=1)
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=SimpleOutput,
            prompt="Generate multiple outputs",
            llm=llm,
            allow_parallel_tool_calls=True,
        )

        results = tools_llm()

        assert isinstance(results, list)
        assert len(results) == 2  # MockLLM returns 2 for parallel
        assert all(isinstance(r, SimpleOutput) for r in results)


# ============================================================================
# Integration Tests: Regular Functions with Mock LLM
# ============================================================================


class TestRegularFunctionsIntegration:
    """Integration tests using regular Python functions with MockLLM.

    Tests the complete flow with various types of regular functions
    as output_cls.
    """

    @pytest.mark.integration
    def test_simple_function_sync(self):
        """Test ToolOrchestratingLLM with simple function (sync).

        Expected: Should execute and return dict from simple_function.
        """
        mock_output = {"result": "TEST", "length": 4}
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=simple_function,
            prompt="Process {text}",
            llm=llm,
        )

        result = tools_llm(text="test")

        assert isinstance(result, dict)
        assert result["result"] == "TEST"
        assert result["length"] == 4

    @pytest.mark.integration
    def test_function_with_multiple_parameters(self):
        """Test function with multiple parameters and defaults.

        Expected: Should handle complex function signatures correctly.
        """
        mock_output = {"operation": "add", "result": 15, "inputs": [10, 5]}
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=complex_function,
            prompt="Calculate {a} and {b}",
            llm=llm,
        )

        result = tools_llm(a=10, b=5)

        assert isinstance(result, dict)
        assert result["operation"] == "add"
        assert result["result"] == 15

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_function(self):
        """Test ToolOrchestratingLLM with async function.

        Expected: Should handle async functions correctly.
        """
        mock_output = {"value": 5, "doubled": 10}
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=async_function,
            prompt="Process value {value}",
            llm=llm,
        )

        result = await tools_llm.acall(value=5)

        assert isinstance(result, dict)
        assert result["value"] == 5
        assert result["doubled"] == 10

    @pytest.mark.integration
    def test_dataclass_factory_function(self):
        """Test function that wraps dataclass creation.

        Expected: Should execute factory function and return dict.
        """
        mock_output = {"name": "test_item", "score": 95.5, "tags": ["tag1", "tag2"]}
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=dataclass_factory,
            prompt="Create item with {name}",
            llm=llm,
        )

        result = tools_llm(name="test_item")

        assert isinstance(result, dict)
        assert result["name"] == "test_item"
        assert result["score"] == 95.5
        assert len(result["tags"]) == 2

    @pytest.mark.integration
    def test_lambda_function(self):
        """Test ToolOrchestratingLLM with lambda function.

        Expected: Should handle lambda functions correctly.
        """
        lambda_func = lambda x, y: {"sum": x + y, "product": x * y}  # noqa: E731
        mock_output = {"sum": 15, "product": 50}
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=lambda_func,
            prompt="Calculate {x} and {y}",
            llm=llm,
        )

        result = tools_llm(x=10, y=5)

        assert isinstance(result, dict)
        assert result["sum"] == 15
        assert result["product"] == 50

    @pytest.mark.integration
    @pytest.mark.xfail(
        reason="CallableTool.from_function() doesn't support callable instances yet"
    )
    def test_callable_class_instance(self):
        """Test ToolOrchestratingLLM with callable class instance.

        Expected: Currently fails because CallableTool.from_function() doesn't support
        callable class instances (only functions). This is a known limitation.
        """
        callable_obj = CallableClass()
        mock_output = {"message": "hello", "reversed": "olleh", "upper": "HELLO"}
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=callable_obj,
            prompt="Process {message}",
            llm=llm,
        )

        result = tools_llm(message="hello")

        assert isinstance(result, dict)
        assert result["message"] == "hello"
        assert result["reversed"] == "olleh"


# ============================================================================
# Integration Tests: Mixed Usage Patterns
# ============================================================================


class TestMixedUsagePatterns:
    """Integration tests for various usage patterns and edge cases."""

    @pytest.mark.integration
    def test_switching_between_pydantic_and_function(self):
        """Test creating multiple instances with different output_cls types.

        Expected: Both should work independently.
        """
        llm = MockLLM()

        # First with Pydantic model
        tools_llm1 = ToolOrchestratingLLM(
            output_cls=SimpleOutput,
            prompt="Test 1",
            llm=llm,
        )

        # Then with function
        tools_llm2 = ToolOrchestratingLLM(
            output_cls=simple_function,
            prompt="Test 2",
            llm=llm,
        )

        # Both should have valid tools
        tool1 = tools_llm1._create_tool()
        tool2 = tools_llm2._create_tool()

        assert isinstance(tool1, CallableTool)
        assert isinstance(tool2, CallableTool)
        assert tool1.metadata.name == "SimpleOutput"
        assert tool2.metadata.name == "simple_function"

    @pytest.mark.integration
    def test_function_with_verbose_mode(self):
        """Test function with verbose logging enabled.

        Expected: Should execute successfully with verbose=True.
        """
        mock_output = {"result": "verbose_test"}
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=simple_function,
            prompt="Test",
            llm=llm,
            verbose=True,
        )

        result = tools_llm(text="test")

        assert isinstance(result, dict)

    @pytest.mark.integration
    def test_function_with_llm_kwargs(self):
        """Test passing additional LLM kwargs with function output_cls.

        Expected: Should pass kwargs to LLM correctly.
        """
        mock_output = {"result": "kwargs_test"}
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=simple_function,
            prompt="Test",
            llm=llm,
        )

        result = tools_llm(
            llm_kwargs={"temperature": 0.7, "max_tokens": 100}, text="test"
        )

        assert isinstance(result, dict)


# ============================================================================
# Integration Tests: Complex Argument Types
# ============================================================================


class TestComplexArgumentTypes:
    """Tests for functions with complex argument types.

    Tests various complex type hints including:
    - List[str], List[int], List[float]
    - Dict[str, Any], Dict[str, int]
    - Optional[T]
    - Union[T1, T2]
    - Tuple types
    - Nested complex types
    """

    @pytest.mark.integration
    def test_function_with_list_str_argument(self):
        """Test function with List[str] argument.

        Expected: Should handle list of strings correctly.
        """

        def process_strings(items: List[str]) -> dict:
            """Process a list of strings.

            Args:
                items: List of string items

            Returns:
                Dictionary with processed results
            """
            return {
                "count": len(items),
                "joined": ", ".join(items),
                "lengths": [len(item) for item in items],
            }

        mock_output = {
            "count": 3,
            "joined": "apple, banana, cherry",
            "lengths": [5, 6, 6],
        }
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=process_strings,
            prompt="Process these items: {items}",
            llm=llm,
        )

        result = tools_llm(items=["apple", "banana", "cherry"])

        assert isinstance(result, dict)
        assert result["count"] == 3
        assert "apple" in result["joined"]

    @pytest.mark.integration
    def test_function_with_list_int_argument(self):
        """Test function with List[int] argument.

        Expected: Should handle list of integers correctly.
        """

        def calculate_stats(numbers: List[int]) -> dict:
            """Calculate statistics on a list of integers.

            Args:
                numbers: List of integer numbers

            Returns:
                Dictionary with calculated statistics
            """
            return {
                "sum": sum(numbers),
                "average": sum(numbers) / len(numbers) if numbers else 0,
                "min": min(numbers) if numbers else 0,
                "max": max(numbers) if numbers else 0,
                "count": len(numbers),
            }

        mock_output = {"sum": 150, "average": 30.0, "min": 10, "max": 50, "count": 5}
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=calculate_stats,
            prompt="Calculate stats for: {numbers}",
            llm=llm,
        )

        result = tools_llm(numbers=[10, 20, 30, 40, 50])

        assert isinstance(result, dict)
        assert result["sum"] == 150
        assert result["count"] == 5

    @pytest.mark.integration
    def test_function_with_list_float_argument(self):
        """Test function with List[float] argument.

        Expected: Should handle list of floats correctly.
        """

        def process_measurements(values: List[float], unit: str = "meters") -> dict:
            """Process a list of measurement values.

            Args:
                values: List of float measurements
                unit: Unit of measurement

            Returns:
                Dictionary with processed measurements
            """
            return {
                "total": sum(values),
                "average": sum(values) / len(values) if values else 0.0,
                "unit": unit,
                "precision": 2,
            }

        mock_output = {
            "total": 45.7,
            "average": 15.23,
            "unit": "meters",
            "precision": 2,
        }
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=process_measurements,
            prompt="Process measurements: {values}",
            llm=llm,
        )

        result = tools_llm(values=[12.5, 15.7, 17.5])

        assert isinstance(result, dict)
        assert "total" in result
        assert "average" in result

    @pytest.mark.integration
    def test_function_with_dict_argument(self):
        """Test function with Dict[str, Any] argument.

        Expected: Should handle dictionary arguments correctly.
        """

        def process_config(config: Dict[str, Any]) -> dict:
            """Process configuration dictionary.

            Args:
                config: Configuration dictionary

            Returns:
                Processed configuration
            """
            return {
                "keys": list(config.keys()),
                "key_count": len(config),
                "has_name": "name" in config,
                "processed": True,
            }

        mock_output = {
            "keys": ["name", "age", "email"],
            "key_count": 3,
            "has_name": True,
            "processed": True,
        }
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=process_config,
            prompt="Process config",
            llm=llm,
        )

        result = tools_llm(
            config={"name": "Alice", "age": 30, "email": "alice@example.com"}
        )

        assert isinstance(result, dict)
        assert result["has_name"] is True
        assert result["key_count"] == 3

    @pytest.mark.integration
    def test_function_with_dict_str_int_argument(self):
        """Test function with Dict[str, int] typed argument.

        Expected: Should handle typed dictionary arguments.
        """

        def analyze_scores(scores: Dict[str, int]) -> dict:
            """Analyze a dictionary of scores.

            Args:
                scores: Dictionary mapping names to scores

            Returns:
                Analysis results
            """
            return {
                "total_students": len(scores),
                "average_score": sum(scores.values()) / len(scores) if scores else 0,
                "highest_score": max(scores.values()) if scores else 0,
                "lowest_score": min(scores.values()) if scores else 0,
            }

        mock_output = {
            "total_students": 3,
            "average_score": 85.0,
            "highest_score": 95,
            "lowest_score": 75,
        }
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=analyze_scores,
            prompt="Analyze these scores",
            llm=llm,
        )

        result = tools_llm(scores={"Alice": 95, "Bob": 80, "Charlie": 75})

        assert isinstance(result, dict)
        assert result["total_students"] == 3

    @pytest.mark.integration
    def test_function_with_optional_argument(self):
        """Test function with Optional[T] argument.

        Expected: Should handle optional arguments correctly.
        """

        def format_name(first: str, last: str, middle: Optional[str] = None) -> dict:
            """Format a person's name.

            Args:
                first: First name
                last: Last name
                middle: Optional middle name

            Returns:
                Formatted name dictionary
            """
            full_name = f"{first} {middle + ' ' if middle else ''}{last}"
            return {
                "first": first,
                "last": last,
                "middle": middle,
                "full_name": full_name.strip(),
                "has_middle": middle is not None,
            }

        mock_output = {
            "first": "John",
            "last": "Doe",
            "middle": "Michael",
            "full_name": "John Michael Doe",
            "has_middle": True,
        }
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=format_name,
            prompt="Format name: {first} {last}",
            llm=llm,
        )

        result = tools_llm(first="John", last="Doe", middle="Michael")

        assert isinstance(result, dict)
        assert result["has_middle"] is True

    @pytest.mark.integration
    def test_function_with_union_argument(self):
        """Test function with Union[str, int] argument.

        Expected: Should handle union type arguments.
        """

        def process_value(value: Union[str, int]) -> dict:
            """Process a value that can be string or int.

            Args:
                value: Either a string or integer

            Returns:
                Processing result
            """
            return {
                "value": str(value),
                "type": type(value).__name__,
                "is_numeric": isinstance(value, int),
                "length": len(str(value)),
            }

        mock_output = {"value": "42", "type": "int", "is_numeric": True, "length": 2}
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=process_value,
            prompt="Process value: {value}",
            llm=llm,
        )

        result = tools_llm(value=42)

        assert isinstance(result, dict)
        assert "type" in result

    @pytest.mark.integration
    def test_function_with_tuple_argument(self):
        """Test function with tuple argument.

        Expected: Should handle tuple arguments correctly.
        """

        def calculate_distance(point1: tuple, point2: tuple) -> dict:
            """Calculate distance between two points.

            Args:
                point1: First point (x, y)
                point2: Second point (x, y)

            Returns:
                Distance calculation result
            """
            import math

            distance = math.sqrt(
                (point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2
            )
            return {"point1": point1, "point2": point2, "distance": distance}

        mock_output = {"point1": (0, 0), "point2": (3, 4), "distance": 5.0}
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=calculate_distance,
            prompt="Calculate distance",
            llm=llm,
        )

        result = tools_llm(point1=(0, 0), point2=(3, 4))

        assert isinstance(result, dict)
        assert "distance" in result

    @pytest.mark.integration
    def test_function_with_nested_list_argument(self):
        """Test function with nested List[List[int]] argument.

        Expected: Should handle nested list types correctly.
        """

        def process_matrix(matrix: List[List[int]]) -> dict:
            """Process a 2D matrix of integers.

            Args:
                matrix: 2D list of integers

            Returns:
                Matrix processing results
            """
            rows = len(matrix)
            cols = len(matrix[0]) if matrix else 0
            flat = [item for row in matrix for item in row]

            return {
                "rows": rows,
                "cols": cols,
                "total_elements": len(flat),
                "sum": sum(flat),
                "dimensions": f"{rows}x{cols}",
            }

        mock_output = {
            "rows": 3,
            "cols": 3,
            "total_elements": 9,
            "sum": 45,
            "dimensions": "3x3",
        }
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=process_matrix,
            prompt="Process matrix",
            llm=llm,
        )

        result = tools_llm(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        assert isinstance(result, dict)
        assert result["total_elements"] == 9

    @pytest.mark.integration
    def test_function_with_nested_dict_argument(self):
        """Test function with nested Dict[str, Dict[str, int]] argument.

        Expected: Should handle nested dictionary types correctly.
        """

        def process_nested_data(data: Dict[str, Dict[str, int]]) -> dict:
            """Process nested dictionary structure.

            Args:
                data: Nested dictionary mapping

            Returns:
                Processing results
            """
            total_keys = len(data)
            total_values = sum(len(v) for v in data.values())

            return {
                "top_level_keys": total_keys,
                "total_nested_values": total_values,
                "keys": list(data.keys()),
                "structure": "nested",
            }

        mock_output = {
            "top_level_keys": 2,
            "total_nested_values": 5,
            "keys": ["user1", "user2"],
            "structure": "nested",
        }
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=process_nested_data,
            prompt="Process data",
            llm=llm,
        )

        result = tools_llm(
            data={
                "user1": {"score": 100, "level": 5},
                "user2": {"score": 85, "level": 4, "bonus": 10},
            }
        )

        assert isinstance(result, dict)
        assert result["structure"] == "nested"

    @pytest.mark.integration
    def test_function_with_list_dict_argument(self):
        """Test function with List[Dict[str, Any]] argument.

        Expected: Should handle list of dictionaries correctly.
        """

        def process_records(records: List[Dict[str, Any]]) -> dict:
            """Process a list of record dictionaries.

            Args:
                records: List of record dictionaries

            Returns:
                Summary of records
            """
            return {
                "record_count": len(records),
                "total_keys": sum(len(r.keys()) for r in records),
                "has_data": len(records) > 0,
                "sample_keys": list(records[0].keys()) if records else [],
            }

        mock_output = {
            "record_count": 3,
            "total_keys": 9,
            "has_data": True,
            "sample_keys": ["name", "age", "city"],
        }
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=process_records,
            prompt="Process records",
            llm=llm,
        )

        result = tools_llm(
            records=[
                {"name": "Alice", "age": 30, "city": "NYC"},
                {"name": "Bob", "age": 25, "city": "LA"},
                {"name": "Charlie", "age": 35, "city": "SF"},
            ]
        )

        assert isinstance(result, dict)
        assert result["record_count"] == 3

    @pytest.mark.integration
    def test_function_with_optional_list_argument(self):
        """Test function with Optional[List[str]] argument.

        Expected: Should handle optional list arguments.
        """

        def process_tags(title: str, tags: Optional[List[str]] = None) -> dict:
            """Process item with optional tags.

            Args:
                title: Item title
                tags: Optional list of tags

            Returns:
                Processed item data
            """
            return {
                "title": title,
                "tags": tags or [],
                "tag_count": len(tags) if tags else 0,
                "has_tags": tags is not None and len(tags) > 0,
            }

        mock_output = {
            "title": "Article",
            "tags": ["python", "ai", "ml"],
            "tag_count": 3,
            "has_tags": True,
        }
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=process_tags,
            prompt="Process item: {title}",
            llm=llm,
        )

        result = tools_llm(title="Article", tags=["python", "ai", "ml"])

        assert isinstance(result, dict)
        assert result["has_tags"] is True

    @pytest.mark.integration
    def test_function_with_complex_nested_types(self):
        """Test function with highly complex nested type annotations.

        Expected: Should handle Dict[str, List[Dict[str, Union[int, str]]]] correctly.
        """

        def process_complex_structure(
            data: Dict[str, List[Dict[str, Union[int, str]]]],
        ) -> dict:
            """Process a complex nested data structure.

            Args:
                data: Complex nested structure

            Returns:
                Analysis of the structure
            """
            total_lists = len(data)
            total_items = sum(len(v) for v in data.values())

            return {
                "total_lists": total_lists,
                "total_items": total_items,
                "complexity": "high",
                "structure_valid": True,
            }

        mock_output = {
            "total_lists": 2,
            "total_items": 4,
            "complexity": "high",
            "structure_valid": True,
        }
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=process_complex_structure,
            prompt="Process complex data",
            llm=llm,
        )

        result = tools_llm(
            data={
                "users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
                "products": [{"id": "p1", "price": 100}, {"id": "p2", "price": 200}],
            }
        )

        assert isinstance(result, dict)
        assert result["structure_valid"] is True

    @pytest.mark.integration
    def test_function_with_mixed_complex_arguments(self):
        """Test function with multiple complex argument types.

        Expected: Should handle mixed complex types in same function.
        """

        def analyze_data(
            names: List[str],
            scores: Dict[str, float],
            metadata: Optional[Dict[str, Any]] = None,
            threshold: float = 0.5,
        ) -> dict:
            """Analyze data with mixed complex types.

            Args:
                names: List of names
                scores: Dictionary of scores
                metadata: Optional metadata dictionary
                threshold: Score threshold

            Returns:
                Analysis results
            """
            return {
                "name_count": len(names),
                "score_count": len(scores),
                "has_metadata": metadata is not None,
                "threshold": threshold,
                "above_threshold": sum(1 for v in scores.values() if v > threshold),
            }

        mock_output = {
            "name_count": 3,
            "score_count": 3,
            "has_metadata": True,
            "threshold": 0.5,
            "above_threshold": 2,
        }
        llm = MockLLM(return_value=mock_output)

        tools_llm = ToolOrchestratingLLM(
            output_cls=analyze_data,
            prompt="Analyze data",
            llm=llm,
        )

        result = tools_llm(
            names=["Alice", "Bob", "Charlie"],
            scores={"Alice": 0.9, "Bob": 0.3, "Charlie": 0.7},
            metadata={"source": "test"},
        )

        assert isinstance(result, dict)
        assert result["above_threshold"] == 2


# ============================================================================
# E2E Tests: Real Ollama Integration
# ============================================================================


class TestOllamaE2E:
    """End-to-end tests with real Ollama server.

    These tests require:
    - Ollama server running
    - llama3.1 model pulled

    Skip if not available using pytest markers.
    """

    @pytest.mark.e2e
    def test_pydantic_model_with_real_ollama(self):
        """Test Pydantic model with real Ollama server.

        Expected: Should generate valid SimpleOutput from LLM.
        """
        from serapeum.llms.ollama import Ollama

        llm = Ollama(model="llama3.1", request_timeout=80)

        tools_llm = ToolOrchestratingLLM(
            output_cls=SimpleOutput,
            prompt="Generate a simple output with value '{text}' and count the words",
            llm=llm,
        )

        result = tools_llm(text="hello world")

        assert isinstance(result, SimpleOutput)
        assert isinstance(result.value, str)
        assert isinstance(result.count, int)

    @pytest.mark.e2e
    def test_function_with_real_ollama(self):
        """Test regular function with real Ollama server.

        Expected: Should generate valid dict output from function via LLM.
        """
        from serapeum.llms.ollama import Ollama

        def extract_info(name: str, age: int, city: str) -> dict:
            """Extract person information."""
            return {
                "name": name,
                "age": age,
                "city": city,
                "summary": f"{name} is {age} years old and lives in {city}",
            }

        llm = Ollama(model="llama3.1", request_timeout=80)

        tools_llm = ToolOrchestratingLLM(
            output_cls=extract_info,
            prompt="Extract information from: {text}",
            llm=llm,
        )

        result = tools_llm(text="John is 30 years old and lives in New York")

        assert isinstance(result, dict)
        assert "name" in result
        assert "age" in result
        assert "city" in result

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_async_function_with_real_ollama(self):
        """Test async function with real Ollama server.

        Expected: Should execute async function via LLM successfully.
        """
        from serapeum.llms.ollama import Ollama

        async def async_processor(text: str, multiplier: int) -> dict:
            """Process text asynchronously."""
            await asyncio.sleep(0.01)
            return {
                "text": text,
                "length": len(text),
                "multiplied": len(text) * multiplier,
            }

        llm = Ollama(model="llama3.1", request_timeout=80)

        tools_llm = ToolOrchestratingLLM(
            output_cls=async_processor,
            prompt="Process this text: {text}",
            llm=llm,
        )

        result = await tools_llm.acall(text="hello")

        assert isinstance(result, dict)
        assert "text" in result
        assert "length" in result


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.unit
    def test_function_returning_non_dict(self):
        """Test function that returns non-dict value.

        Expected: Should still work as CallableTool.from_function handles it.
        """

        def return_string() -> str:
            return "just a string"

        llm = MockLLM(return_value="test_string")

        tools_llm = ToolOrchestratingLLM(
            output_cls=return_string,
            prompt="Test",
            llm=llm,
        )

        # Should create tool successfully
        tool = tools_llm._create_tool()
        assert isinstance(tool, CallableTool)

    @pytest.mark.unit
    def test_function_with_no_parameters(self):
        """Test function with no parameters.

        Expected: Should handle parameter-less functions.
        """

        def no_params() -> dict:
            return {"message": "no params"}

        llm = MockLLM(return_value={"message": "no params"})

        tools_llm = ToolOrchestratingLLM(
            output_cls=no_params,
            prompt="Generate output",
            llm=llm,
        )

        tool = tools_llm._create_tool()
        assert isinstance(tool, CallableTool)

    @pytest.mark.unit
    def test_function_with_only_kwargs(self):
        """Test function with **kwargs only.

        Expected: Should handle flexible parameter functions.
        """

        def flexible_func(**kwargs) -> dict:
            return kwargs

        llm = MockLLM(return_value={"a": 1, "b": 2})

        tools_llm = ToolOrchestratingLLM(
            output_cls=flexible_func,
            prompt="Test",
            llm=llm,
        )

        tool = tools_llm._create_tool()
        assert isinstance(tool, CallableTool)

    @pytest.mark.unit
    def test_pydantic_model_with_validators(self):
        """Test Pydantic model with field validators.

        Expected: Should handle models with validators correctly.
        """
        from pydantic import field_validator

        class ValidatedModel(BaseModel):
            email: str
            age: int

            @field_validator("email")
            @classmethod
            def validate_email(cls, v):
                if "@" not in v:
                    raise ValueError("Invalid email")
                return v

            @field_validator("age")
            @classmethod
            def validate_age(cls, v):
                if v < 0 or v > 150:
                    raise ValueError("Invalid age")
                return v

        llm = MockLLM()

        tools_llm = ToolOrchestratingLLM(
            output_cls=ValidatedModel,
            prompt="Test",
            llm=llm,
        )

        tool = tools_llm._create_tool()
        assert isinstance(tool, CallableTool)
        assert tool.metadata.name == "ValidatedModel"
