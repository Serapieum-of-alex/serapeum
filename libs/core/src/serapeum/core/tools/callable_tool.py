"""LLM function-calling functionalities."""

import asyncio
import inspect
from typing import Any, Awaitable, Callable, Type, TypeVar

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from serapeum.core.base.llms.types import Audio, ChunkType, Image, TextChunk
from serapeum.core.tools.types import (
    ArgumentCoercer,
    AsyncBaseTool,
    ToolMetadata,
    ToolOutput,
)
from serapeum.core.tools.convert import Docstring, FunctionConverter
from serapeum.core.tools.async_utils import asyncio_run

AsyncCallable = Callable[..., Awaitable[Any]]

Model = TypeVar("Model", bound=BaseModel)


class SyncAsyncConverter:
    """Sync/async function converter for tool callables."""

    def __init__(self, func: Callable[..., Any]) -> None:
        """Initialize SyncAsyncConverter with a callable."""
        if not callable(func):
            raise ValueError("func must be a callable")

        if self.is_async(func):
            self.async_func = func
            self.sync_func = self.async_to_sync(func)
        else:
            self.sync_func = func
            self.async_func = self.to_async(func)

    @staticmethod
    def is_async(func: Callable[..., Any] | Callable[..., Awaitable[Any]]) -> bool:
        return inspect.iscoroutinefunction(func)

    @staticmethod
    def to_async(fn: Callable[..., Any]) -> AsyncCallable:
        """Wrap a synchronous callable so it can be awaited.

        The returned coroutine function offloads the synchronous work to the event
        loop's default executor using ``loop.run_in_executor(None, ...)``.

        Args:
            fn (Callable[..., Any]):
                A regular, synchronous callable to be executed in a thread pool.

        Returns:
            AsyncCallable: An ``async`` wrapper that accepts the same positional and
            keyword arguments as ``func`` and returns its result when awaited.

        Raises:
            RuntimeError: If called when no event loop is running. Use
                ``asyncio.run(...)`` or ensure you're inside an async context when
                awaiting the wrapper.

        Examples:
            - Basic usage with integers
                ```python
                >>> import asyncio
                >>> from serapeum.core.tools.callable_tool import SyncAsyncConverter
                >>> def add(x: int, y: int) -> int:
                ...     return x + y
                >>> converter = SyncAsyncConverter(add)
                >>> async_add = converter.async_func
                >>> print(asyncio.run(async_add(2, 3)))
                5

                ```
            - Works with arbitrary return types
                ```python
                >>> import asyncio
                >>> from serapeum.core.tools.callable_tool import SyncAsyncConverter
                >>> def greet(name: str) -> str:
                ...     return f"Hello, {name}!"
                >>> converter = SyncAsyncConverter(greet)
                >>> async_greet = converter.async_func
                >>> print(asyncio.run(async_greet("Alice")))
                Hello, Alice!

                ```

        See Also:
            - async_to_sync: Convert an async callable into a blocking callable.
        """

        async def _async_wrapped_fn(*args: Any, **kwargs: Any) -> Any:
            loop = asyncio.get_running_loop()
            # offload a blocking function to a thread pool.
            return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

        return _async_wrapped_fn

    @staticmethod
    def async_to_sync(func_async: AsyncCallable) -> Callable[..., Any]:
        """Wrap an async callable so it can be used from synchronous code.

        This wrapper runs the coroutine to completion using
        :func:`serapeum.core.tools.async_utils.asyncio_run`, which handles common
        environments (including notebooks with existing loops) and provides a clear
        error message for nested event loops.

        Args:
            func_async (AsyncCallable):
                An async function to be executed in a blocking manner.

        Returns:
            Callable: A synchronous function that blocks until the coroutine
            completes and returns its result.

        Raises:
            RuntimeError: If the environment disallows running a new or nested event
                loop. See the error message for mitigation strategies.

        Examples:
            - Basic usage
                ```python
                >>> from serapeum.core.tools.callable_tool import SyncAsyncConverter
                >>> async def mul(x: int, y: int) -> int:
                ...     return x * y
                >>> converter = SyncAsyncConverter(mul)
                >>> mul_sync = converter.sync_func
                >>> print(mul_sync(4, 5))
                20

                ```
            - Wrapping async functions that return strings
                ```python
                >>> from serapeum.core.tools.callable_tool import SyncAsyncConverter
                >>> async def greet(name: str) -> str:
                ...     return f"Hello, {name}!"
                >>> converter = SyncAsyncConverter(greet)
                >>> greet_sync = converter.sync_func
                >>> print(greet_sync("Bob"))
                Hello, Bob!

                ```

        See Also:
            - sync_to_async: Convert a sync callable into an awaitable callable.
        """

        def _sync_wrapped_fn(*args: Any, **kwargs: Any) -> Any:
            return asyncio_run(func_async(*args, **kwargs))  # type: ignore[arg-type]

        return _sync_wrapped_fn


class CallableTool(AsyncBaseTool):
    """Adapter that turns a Python callable into a Tool that can be called from LLMs.

    ``CallableTool`` wraps either a synchronous or asynchronous function and
    normalizes input/output into the standard :class:`~serapeum.core.tools.types.ToolOutput`.
    It also carries :class:`~serapeum.core.tools.types.ToolMetadata` used by
    LLM function-calling interfaces.

    Use the constructor when you already have a :class:`ToolMetadata` instance,
    or :meth:`CallableTool.from_defaults` to automatically derive metadata and an
    input schema from the target function's signature and docstring.

    Examples:
        - Wrap a synchronous function
            ```python
            >>> from serapeum.core.tools.callable_tool import CallableTool
            >>> from serapeum.core.tools.types import ToolMetadata
            >>> def greet(name: str) -> str:
            ...     return f"Hello, {name}!"
            >>> meta = ToolMetadata(name="greet", description="Greets a person by name.")
            >>> tool = CallableTool(func=greet, metadata=meta)
            >>> out = tool("World")  # __call__ delegates to .call()
            >>> print(out.content)
            Hello, World!

            ```
        - Wrap an async function
            ```python
            >>> import asyncio
            >>> from serapeum.core.tools.callable_tool import CallableTool
            >>> from serapeum.core.tools.types import ToolMetadata
            >>> async def add(x: int, y: int) -> int:
            ...     return x + y
            >>> tool = CallableTool(func=add, metadata=ToolMetadata(name="add", description="Add two ints"))
            >>> out = asyncio.run(tool.acall(2, 3))  # call() also works via sync adapter
            >>> print(out.content)
            5

            ```
        - Provide default arguments
            ```python
            >>> from serapeum.core.tools.callable_tool import CallableTool
            >>> from serapeum.core.tools.types import ToolMetadata
            >>> def power(base: int, exp: int = 2) -> int:
            ...     return base ** exp
            >>> tool = CallableTool(
            ...     func=power,
            ...     metadata=ToolMetadata(name="power", description="Exponentiation"),
            ...     default_arguments={"exp": 3},
            ... )
            >>> out = tool(2)  # uses exp=3 by default, can be overridden
            >>> print(out.content)
            8

            ```

    See Also:
        - serapeum.core.tools.types.ToolOutput: Standardized tool output container.
        - serapeum.core.tools.types.ToolMetadata: Metadata used for function calling.
        - CallableTool.from_defaults: Build a tool with inferred metadata/schema.
    """

    def __init__(
        self,
        func: Callable[..., Any] | AsyncCallable | None,
        metadata: ToolMetadata | None = None,
        default_arguments: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a CallableTool.

        Provide either a synchronous function via ``func`` or an asynchronous one
        via ``async_fn``. If only ``func`` is provided and it is synchronous, an
        async adapter is created automatically; if ``func`` is already async, a
        sync adapter is created for ``call(...)``.

        Args:
            func (Optional[Callable[..., Any]]):
                Synchronous or asynchronous callable to wrap. Mutually exclusive
                with ``async_fn``.
            metadata (ToolMetadata | None):
                Required metadata describing the tool. Use
                :meth:`CallableTool.from_defaults` if you prefer automatic
                metadata derivation.
            default_arguments (dict[str, Any] | None):
                Default keyword arguments that will be merged into each call,
                allowing you to pre-configure parameters.

        Raises:
            ValueError: If neither ``func`` nor ``async_fn`` is provided.
            ValueError: If ``metadata`` is not provided.

        Examples:
            - Synchronous function with metadata
                ```python
                >>> from serapeum.core.tools.callable_tool import CallableTool
                >>> from serapeum.core.tools.types import ToolMetadata
                >>> def echo(text: str) -> str:
                ...     return text
                >>> tool = CallableTool(func=echo, metadata=ToolMetadata(name="echo", description="Echo text"))
                >>> print(tool("hi").content)
                hi

                ```
            - Asynchronous function with default arguments
                ```python
                >>> import asyncio
                >>> from serapeum.core.tools.callable_tool import CallableTool
                >>> from serapeum.core.tools.types import ToolMetadata
                >>> async def add(a: int, b: int) -> int:
                ...     return a + b
                >>> tool = CallableTool(
                ...     func=add,
                ...     metadata=ToolMetadata(name="add", description="Add ints"),
                ...     default_arguments={"b": 10},
                ... )
                >>> print(asyncio.run(tool.acall(5)).content)
                15

                ```
        """
        # Handle function (sync and async)
        self._input_func = func
        sync_async_converter = SyncAsyncConverter(func)

        self._async_func = sync_async_converter.async_func
        self._sync_func = sync_async_converter.sync_func

        if metadata is None:
            raise ValueError("metadata must be provided")

        self._metadata = metadata
        self.default_arguments = default_arguments or {}

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any] | AsyncCallable | None,
        name: str | None = None,
        description: str | None = None,
        return_direct: bool = False,
        tool_schema: Type[BaseModel] | None = None,
        tool_metadata: ToolMetadata | None = None,
        default_arguments: dict[str, Any] | None = None,
    ) -> "CallableTool":
        """Construct a ``CallableTool`` and infer metadata/schema when needed.

        If ``tool_metadata`` is not provided, this method derives a
        :class:`ToolMetadata` instance from the provided ``func``/``async_fn`` by
        inspecting its signature, type hints, and docstring. If
        ``tool_schema`` is not provided, a Pydantic model is built to represent
        the function's input schema. Where possible, parameter descriptions are
        extracted from the function's docstring via
        :meth:`CallableTool.extract_param_docs`.

        Args:
            func (Optional[Callable[..., Any]]):
                Synchronous or Asynchronous function to wrap.
            name (str | None):
                Override for the tool name. Defaults to the function's
                ``__name__``.
            description (str | None):
                Override for the tool description. If omitted, a description is
                constructed using the function signature and the first
                non-empty line of its docstring, if available.
            return_direct (bool):
                Whether the tool's content should be returned directly by some
                orchestrators. Defaults to ``False``.
            tool_schema (Type[BaseModel] | None):
                Optional Pydantic model to use as the input schema. If omitted,
                one is created from the function signature.
            tool_metadata (ToolMetadata | None):
                Provide explicit metadata. If given, it is used as-is and no
                inference occurs.
            default_arguments (dict[str, Any] | None):
                Default keyword arguments to merge into every call.

        Returns:
            CallableTool: A tool wrapping the provided callable.

        Examples:
            - Infer metadata and schema from a sync function
                ```python
                >>> from serapeum.core.tools.callable_tool import CallableTool
                >>> def add(a: int, b: int) -> int:
                ...     '''Add two integers.
                ...
                ...     Args:
                ...         a (int): First integer.
                ...         b (int): Second integer.
                ...     '''
                ...     return a + b
                >>> tool = CallableTool.from_function(func=add)
                >>> # Name falls back to function name; calling works as usual
                >>> print(tool.metadata.get_name())
                add
                >>> print(tool(2, 3).content)
                5

                ```
            - Provide your own metadata and defaults
                ```python
                >>> from serapeum.core.tools.types import ToolMetadata
                >>> from serapeum.core.tools.callable_tool import CallableTool
                >>> def echo(text: str) -> str:
                ...     return text
                >>> meta = ToolMetadata(name="echo", description="Echo text")
                >>> tool = CallableTool.from_function(func=echo, tool_metadata=meta, default_arguments={})
                >>> print(tool("ok").content)
                ok

                ```

        See Also:
            - DocString.extract_param_docs: Parses parameter docs from a function docstring.
        """
        default_arguments = default_arguments or {}

        if tool_metadata is None:
            if func is None:
                raise ValueError("func must be provided")

            name = name or func.__name__
            func_signature = inspect.signature(func)

            # get the docstring from the function
            docstring = Docstring(func)
            param_docs, _ = docstring.extract_param_docs()

            # Replace default values to be required
            func_signature = func_signature.replace(
                parameters=[
                    (
                        param.replace(default=inspect.Parameter.empty)
                        if isinstance(param.default, FieldInfo)
                        else param
                    )
                    for param in func_signature.parameters.values()
                ]
            )
            docstring.signature = func_signature

            # Build enriched description using param_docs
            if description is None:
                description = docstring.get_short_summary_line()

            # get the tool_schema
            if tool_schema is None:

                function = FunctionConverter(
                    f"{name}",
                    func,
                    additional_fields=None,
                    ignore_fields=None,
                )
                tool_schema = function.to_schema()
                if tool_schema is not None and param_docs:
                    for param_name, field in tool_schema.model_fields.items():
                        if not field.description and param_name in param_docs:
                            field.description = param_docs[param_name].strip()

            tool_metadata = ToolMetadata(
                name=name,
                description=description,
                tool_schema=tool_schema,
                return_direct=return_direct,
            )
        return cls(
            func=func,
            metadata=tool_metadata,
            default_arguments=default_arguments,
        )

    @classmethod
    def from_model(cls, output_cls: Type[Model]) -> "CallableTool":
        """Create a callable tool from a Pydantic model class.

        Converts a Pydantic BaseModel class into a CallableTool that can be used
        with LLM function calling. The tool's metadata (name, description, schema)
        is extracted from the model's JSON schema.

        Args:
            output_cls (Type[Model]):
                A Pydantic BaseModel subclass that defines the structure of the tool's output.
                The model's schema will be used to generate the tool's metadata and validation.

        Returns:
            CallableTool:
                A callable tool instance that wraps the Pydantic model. The tool can be invoked with keyword arguments
                matching the model's fields, and will return an instance of the output_cls.

        Example:
            - Create a tool from a simple Pydantic model:
                ```python
                >>> from pydantic import BaseModel, Field
                >>> from serapeum.core.tools.callable_tool import CallableTool
                >>>
                >>> class UserInfo(BaseModel):
                ...     '''User information model.'''
                ...     name: str = Field(description="User's full name")
                ...     age: int = Field(description="User's age in years")
                >>>
                >>> tool = CallableTool.from_model(UserInfo)
                >>> tool.metadata.name
                'UserInfo'
                >>> isinstance(tool, CallableTool)
                True

                ```

            - Use the tool to create model instances:
                ```python
                >>> user = tool.sync_func(name="Alice Smith", age=30)
                >>> user.name
                'Alice Smith'
                >>> user.age
                30
                >>> isinstance(user, UserInfo)
                True

                ```

            - Tool with optional fields:
                ```python
                >>> class Product(BaseModel):
                ...     '''Product information.'''
                ...     name: str
                ...     price: float
                ...     in_stock: bool = True
                >>>
                >>> product_tool = CallableTool.from_model(Product)
                >>> product = product_tool.fn(name="Widget", price=29.99)
                >>> product.in_stock
                True

                ```

            - Tool with nested models:
                ```python
                >>> class Address(BaseModel):
                ...     '''Address information.'''
                ...     street: str
                ...     city: str
                >>>
                >>> class Customer(BaseModel):
                ...     '''Customer with address.'''
                ...     name: str
                ...     address: Address
                >>>
                >>> customer_tool = CallableTool.from_model(Customer)
                >>> customer = customer_tool.sync_func(
                ...     name="Bob",
                ...     address={"street": "123 Main St", "city": "Boston"}
                ... )
                >>> customer.address.city
                'Boston'

                ```

        See Also:
            - CallableTool.from_function: The underlying factory method used
            - ToolOrchestratingLLM: Uses this function to create tools for LLM calls
            - _parse_tool_outputs: Parses outputs from tools created by this function
        """
        schema = output_cls.model_json_schema()
        schema_description = schema.get("description", None)
        model_doc = (output_cls.__doc__ or "").strip()
        # Prefer the model's own description/docstring; fall back to a concise default
        description = (
            schema_description
            or model_doc
            or f"Create an instance of {schema['title']}."
        )

        # NOTE: this does not specify the schema in the function signature,
        # so instead we'll directly provide it in the tool_schema in the ToolMetadata
        def model_fn(**kwargs: Any) -> BaseModel:
            """Model function."""
            coercer = ArgumentCoercer(tool_schema=output_cls)
            coerced_kwargs = coercer.coerce(kwargs)
            return output_cls(**coerced_kwargs)

        return cls.from_function(
            func=model_fn,
            name=schema["title"],
            description=description,
            tool_schema=output_cls,
        )

    @property
    def metadata(self) -> ToolMetadata:
        """Return the tool's metadata.

        Returns:
            ToolMetadata: The metadata object supplied or inferred for this tool.

        Examples:
            - Access the tool name
                ```python
                >>> from serapeum.core.tools.callable_tool import CallableTool
                >>> from serapeum.core.tools.types import ToolMetadata
                >>> def noop(x: int) -> int:
                ...     return x
                >>> tool = CallableTool(func=noop, metadata=ToolMetadata(name="noop", description="No-op"))
                >>> print(tool.metadata.get_name())
                noop

                ```
        """
        return self._metadata

    @property
    def sync_func(self) -> Callable[..., Any]:
        """Return the synchronous wrapper for the underlying callable.

        If the original callable was async, this is a blocking adapter created by
        :func:`async_to_sync`.

        Returns:
            Callable[..., Any]: A callable that can be invoked from sync code.

        Examples:
            - Invoke the underlying function directly
                ```python
                >>> from serapeum.core.tools.callable_tool import CallableTool
                >>> from serapeum.core.tools.types import ToolMetadata
                >>> def add(a: int, b: int) -> int:
                ...     return a + b
                >>> tool = CallableTool(func=add, metadata=ToolMetadata(name="add", description="Add"))
                >>> print(tool.sync_func(2, 3))
                5

                ```
        """
        return self._sync_func

    @property
    def async_func(self) -> AsyncCallable:
        """Return the asynchronous wrapper for the underlying callable.

        If the original callable was synchronous, this is an adapter created by
        :func:`sync_to_async` that runs the function in a thread pool.

        Returns:
            AsyncCallable: A coroutine function that mirrors the wrapped
            callable's signature.

        Examples:
            - Await the async wrapper
                ```python
                >>> import asyncio
                >>> from serapeum.core.tools.callable_tool import CallableTool
                >>> from serapeum.core.tools.types import ToolMetadata
                >>> def square(x: int) -> int:
                ...     return x * x
                >>> tool = CallableTool(func=square, metadata=ToolMetadata(name="square", description="Square"))
                >>> print(asyncio.run(tool.async_func(4)))
                16

                ```
        """
        return self._async_func

    @property
    def input_func(self) -> Callable[..., Any] | AsyncCallable:
        """Return the original callable that was wrapped.

        Returns:
            Callable[..., Any] | AsyncCallable: The underlying function as
            provided to the constructor.

        Raises:
            ValueError: If the tool was improperly initialized and the real
                function is missing (should not occur when using the public
                constructor).

        Examples:
            - Inspect whether the wrapped function is async
                ```python
                >>> import inspect
                >>> from serapeum.core.tools.callable_tool import CallableTool
                >>> from serapeum.core.tools.types import ToolMetadata
                >>> async def afunc(x: int) -> int:
                ...     return x
                >>> tool = CallableTool(func=afunc, metadata=ToolMetadata(name="afunc", description="Async"))
                >>> print(inspect.iscoroutinefunction(tool.input_func))
                True

                ```
        """
        if self._input_func is None:
            raise ValueError("Real function is not set!")

        return self._input_func

    @staticmethod
    def _parse_tool_output(raw_output: Any) -> list[ChunkType]:
        """Normalize raw function output into a list of content chunks.

        The following conversions are performed:
        - A single :class:`TextChunk`, :class:`Image`, or :class:`Audio` becomes
          a single-item list.
        - A list composed entirely of those chunk types is returned as-is.
        - Any other value is coerced to a :class:`TextChunk` via ``str(value)``.

        Args:
            raw_output (Any): The raw value returned by the wrapped callable.

        Returns:
            list[ChunkType]: A list of content chunks suitable for
            :class:`~serapeum.core.tools.types.ToolOutput`.

        Examples:
            - String is converted into a TextChunk
                ```python
                >>> from serapeum.core.tools.callable_tool import CallableTool
                >>> chunks = CallableTool._parse_tool_output("hello")
                >>> print(type(chunks[0]).__name__, getattr(chunks[0], 'content', None))
                TextChunk hello

                ```
            - List of TextChunk values is preserved
                ```python
                >>> from serapeum.core.base.llms.types import TextChunk
                >>> from serapeum.core.tools.callable_tool import CallableTool
                >>> lst = [TextChunk(content="a"), TextChunk(content="b")]
                >>> chunks = CallableTool._parse_tool_output(lst)
                >>> print(len(chunks), type(chunks[1]).__name__)
                2 TextChunk

                ```
        """
        if isinstance(raw_output, (TextChunk, Image, Audio)):
            vals = [raw_output]
        elif isinstance(raw_output, list) and all(
            isinstance(item, (TextChunk, Image, Audio)) for item in raw_output
        ):
            vals = raw_output
        else:
            vals = [TextChunk(content=str(raw_output))]
        return vals

    def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Invoke the tool (synchronous shorthand).

        This is a convenience that merges ``default_arguments`` with ``kwargs``
        and forwards the call to :meth:`CallableTool.call`.

        Args:
            *args (Any): Positional arguments to pass to the wrapped callable.
            **kwargs (Any): Keyword arguments to pass to the wrapped callable.
                These are merged over ``default_arguments`` (explicit values win).

        Returns:
            ToolOutput: A standardized output container with content chunks and
            raw input/output.

        Raises:
            Exception: Any exception raised by the underlying callable is
                propagated.

        Examples:
            - Delegation to call()
                ```python
                >>> from serapeum.core.tools.callable_tool import CallableTool
                >>> from serapeum.core.tools.types import ToolMetadata
                >>> def add(a: int, b: int) -> int:
                ...     return a + b
                >>> tool = CallableTool(func=add, metadata=ToolMetadata(name="add", description="Add"))
                >>> out = tool(2, 3)
                >>> print(out.content)
                5

                ```
            - Default arguments are merged
                ```python
                >>> from serapeum.core.tools.callable_tool import CallableTool
                >>> from serapeum.core.tools.types import ToolMetadata
                >>> def greet(name: str, punct: str = "!") -> str:
                ...     return f"Hello, {name}{punct}"
                >>> tool = CallableTool(
                ...     func=greet,
                ...     metadata=ToolMetadata(name="greet", description="Greet"),
                ...     default_arguments={"punct": "!!"},
                ... )
                >>> print(tool("Ada").content)
                Hello, Ada!!
                >>> print(tool("Ada", punct=".").content)
                Hello, Ada.

                ```
        """
        all_kwargs = {**self.default_arguments, **kwargs}
        return self.call(*args, **all_kwargs)

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Execute the wrapped callable synchronously.

        Merges ``default_arguments`` with the supplied ``kwargs``, invokes the
        synchronous wrapper (:attr:`CallableTool.func`), normalizes the result into
        content chunks, and returns a :class:`ToolOutput`.

        Args:
            *args (Any): Positional arguments to pass to the function.
            **kwargs (Any): Keyword arguments to pass to the function. Values
                here override any overlapping keys in ``default_arguments``.

        Returns:
            ToolOutput: A container with parsed chunks, tool name, raw input, and
            raw output.

        Raises:
            Exception: Any exception raised by the wrapped callable is propagated
                unchanged.

        Examples:
            - Typical usage
                ```python
                >>> from serapeum.core.tools.callable_tool import CallableTool
                >>> from serapeum.core.tools.types import ToolMetadata
                >>> def concat(a: str, b: str) -> str:
                ...     return a + b
                >>> tool = CallableTool(func=concat, metadata=ToolMetadata(name="concat", description="Concat"))
                >>> print(tool.call("a", "b").content)
                ab

                ```
            - Overriding defaults
                ```python
                >>> from serapeum.core.tools.callable_tool import CallableTool
                >>> from serapeum.core.tools.types import ToolMetadata
                >>> def greet(name: str, punct: str = "!") -> str:
                ...     return f"Hello, {name}{punct}"
                >>> tool = CallableTool(
                ...     func=greet,
                ...     metadata=ToolMetadata(name="greet", description="Greet"),
                ...     default_arguments={"punct": "?"},
                ... )
                >>> print(tool.call("Ada").content)
                Hello, Ada?
                >>> print(tool.call("Ada", punct=".").content)
                Hello, Ada.

                ```
        """
        all_kwargs = {**self.default_arguments, **kwargs}

        raw_output = self._sync_func(*args, **all_kwargs)

        # Parse tool output into content chunks
        output_blocks = self._parse_tool_output(raw_output)

        # Default ToolOutput based on the raw output
        default_output = ToolOutput(
            chunks=output_blocks,
            tool_name=self.metadata.get_name(),
            raw_input={"args": args, "kwargs": all_kwargs},
            raw_output=raw_output,
        )

        return default_output

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Execute the wrapped callable asynchronously.

        Uses the asynchronous wrapper (:attr:`CallableTool.async_fn`) and returns
        a :class:`ToolOutput` with normalized content chunks.

        Args:
            *args (Any): Positional arguments for the function.
            **kwargs (Any): Keyword arguments for the function.

        Returns:
            ToolOutput: The standardized output container.

        Raises:
            Exception: Any exception from the coroutine is propagated unchanged.

        Examples:
            - Typical usage with asyncio.run
                ```python
                >>> import asyncio
                >>> from serapeum.core.tools.callable_tool import CallableTool
                >>> from serapeum.core.tools.types import ToolMetadata
                >>> async def add(a: int, b: int) -> int:
                ...     return a + b
                >>> tool = CallableTool(func=add, metadata=ToolMetadata(name="add", description="Add"))
                >>> out = asyncio.run(tool.acall(2, 3))
                >>> print(out.content)
                5

                ```
        """
        all_kwargs = {**self.default_arguments, **kwargs}

        raw_output = await self._async_func(*args, **all_kwargs)

        # Parse tool output into content chunks
        output_blocks = self._parse_tool_output(raw_output)

        # Default ToolOutput based on the raw output
        default_output = ToolOutput(
            chunks=output_blocks,
            tool_name=self.metadata.get_name(),
            raw_input={"args": args, "kwargs": all_kwargs},
            raw_output=raw_output,
        )

        return default_output
