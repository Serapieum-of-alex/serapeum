"""LLM function-calling functionalities."""

import asyncio
import inspect
from typing import Any, Awaitable, Callable, Optional, Type, Dict, Union, List, Tuple
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from serapeum.core.utils.async_utils import asyncio_run
from serapeum.core.tools.models import AsyncBaseTool, ToolMetadata, ToolOutput
from serapeum.core.base.llms.models import (
    TextChunk,
    Image,
    Audio,
    ChunkType,
)
from serapeum.core.tools.utils import FunctionConverter, Docstring

AsyncCallable = Callable[..., Awaitable[Any]]

class SyncAsyncConverter:
    def __init__(self, func: Callable[..., Any]) -> None:
        if not isinstance(func, Callable):
            raise ValueError("func must be a callable")

        if self.is_async(func):
            self.async_func = func
            self.sync_func = self.async_to_sync(func)
        else:
            self.sync_func = func
            self.async_func = self.to_async(func)

    @staticmethod
    def is_async(func) -> bool:
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
    def async_to_sync(func_async: AsyncCallable) -> Callable:
        """Wrap an async callable so it can be used from synchronous code.

        This wrapper runs the coroutine to completion using
        :func:`serapeum.core.utils.async_utils.asyncio_run`, which handles common
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
    normalizes input/output into the standard :class:`~serapeum.core.tools.models.ToolOutput`.
    It also carries :class:`~serapeum.core.tools.models.ToolMetadata` used by
    LLM function-calling interfaces.

    Use the constructor when you already have a :class:`ToolMetadata` instance,
    or :meth:`CallableTool.from_defaults` to automatically derive metadata and an
    input schema from the target function's signature and docstring.

    Examples:
        - Wrap a synchronous function
            ```python
            >>> from serapeum.core.tools.callable_tool import CallableTool
            >>> from serapeum.core.tools.models import ToolMetadata
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
            >>> from serapeum.core.tools.models import ToolMetadata
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
            >>> from serapeum.core.tools.models import ToolMetadata
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
        - serapeum.core.tools.models.ToolOutput: Standardized tool output container.
        - serapeum.core.tools.models.ToolMetadata: Metadata used for function calling.
        - CallableTool.from_defaults: Build a tool with inferred metadata/schema.
    """

    def __init__(
        self,
        func: Optional[Callable[..., Any] | AsyncCallable],
        metadata: Optional[ToolMetadata] = None,
        default_arguments: Optional[Dict[str, Any]] = None,
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
            metadata (Optional[ToolMetadata]):
                Required metadata describing the tool. Use
                :meth:`CallableTool.from_defaults` if you prefer automatic
                metadata derivation.
            async_fn (Optional[AsyncCallable]):
                Async callable to wrap. Mutually exclusive with ``func``.
            default_arguments (Optional[Dict[str, Any]]):
                Default keyword arguments that will be merged into each call,
                allowing you to pre-configure parameters.

        Raises:
            ValueError: If neither ``func`` nor ``async_fn`` is provided.
            ValueError: If ``metadata`` is not provided.

        Examples:
            - Synchronous function with metadata
                ```python
                >>> from serapeum.core.tools.callable_tool import CallableTool
                >>> from serapeum.core.tools.models import ToolMetadata
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
                >>> from serapeum.core.tools.models import ToolMetadata
                >>> async def add(a: int, b: int) -> int:
                ...     return a + b
                >>> tool = CallableTool(func=add, metadata=ToolMetadata(name="add", description="Add ints"), default_arguments={"b": 10})
                >>> print(asyncio.run(tool.acall(5)).content)
                15

                ```
        """
        # Handle function (sync and async)
        self._real_fn = func
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
        func: Optional[Callable[..., Any] | AsyncCallable],
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        tool_schema: Optional[Type[BaseModel]] = None,
        tool_metadata: Optional[ToolMetadata] = None,
        default_arguments: Optional[Dict[str, Any]] = None,
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
                Synchronous function to wrap. Mutually exclusive with
                ``async_fn``.
            name (Optional[str]):
                Override for the tool name. Defaults to the function's
                ``__name__``.
            description (Optional[str]):
                Override for the tool description. If omitted, a description is
                constructed using the function signature and the first
                non-empty line of its docstring, if available.
            return_direct (bool):
                Whether the tool's content should be returned directly by some
                orchestrators. Defaults to ``False``.
            tool_schema (Optional[Type[BaseModel]]):
                Optional Pydantic model to use as the input schema. If omitted,
                one is created from the function signature.
            async_fn (Optional[AsyncCallable]):
                Asynchronous function to wrap. Mutually exclusive with ``func``.
            tool_metadata (Optional[ToolMetadata]):
                Provide explicit metadata. If given, it is used as-is and no
                inference occurs.
            default_arguments (Optional[Dict[str, Any]]):
                Default keyword arguments to merge into every call.

        Returns:
            CallableTool: A tool wrapping the provided callable.

        Raises:
            AssertionError: If neither ``func`` nor ``async_fn`` is provided.

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
                >>> from serapeum.core.tools.models import ToolMetadata
                >>> from serapeum.core.tools.callable_tool import CallableTool
                >>> def echo(text: str) -> str:
                ...     return text
                >>> meta = ToolMetadata(name="echo", description="Echo text")
                >>> tool = CallableTool.from_function(func=echo, tool_metadata=meta, default_arguments={})
                >>> print(tool("ok").content)
                ok

                ```

        See Also:
            - CallableTool.extract_param_docs: Parses parameter docs from a function docstring.
        """
        default_arguments = default_arguments or {}

        if tool_metadata is None:
            if func is None:
                raise ValueError("func must be provided")
            name = name or func.__name__
            fn_sig = inspect.signature(func)

            # get the docstring from the function
            docstring = Docstring(func)
            param_docs, _ = docstring.extract_param_docs()

            # Replace default values to be required
            fn_sig = fn_sig.replace(
                parameters=[
                    (
                        param.replace(default=inspect.Parameter.empty)
                        if isinstance(param.default, FieldInfo)
                        else param
                    )
                    for param in fn_sig.parameters.values()
                ]
            )
            docstring.signature = fn_sig

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


    @property
    def metadata(self) -> ToolMetadata:
        """Return the tool's metadata.

        Returns:
            ToolMetadata: The metadata object supplied or inferred for this tool.

        Examples:
            - Access the tool name
                ```python
                >>> from serapeum.core.tools.callable_tool import CallableTool
                >>> from serapeum.core.tools.models import ToolMetadata
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
                >>> from serapeum.core.tools.models import ToolMetadata
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
                >>> from serapeum.core.tools.models import ToolMetadata
                >>> def square(x: int) -> int:
                ...     return x * x
                >>> tool = CallableTool(func=square, metadata=ToolMetadata(name="square", description="Square"))
                >>> print(asyncio.run(tool.async_func(4)))
                16

                ```
        """
        return self._async_func

    @property
    def real_fn(self) -> Union[Callable[..., Any], AsyncCallable]:
        """Return the original callable that was wrapped.

        Returns:
            Union[Callable[..., Any], AsyncCallable]: The underlying function as
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
                >>> from serapeum.core.tools.models import ToolMetadata
                >>> async def afunc(x: int) -> int:
                ...     return x
                >>> tool = CallableTool(func=afunc, metadata=ToolMetadata(name="afunc", description="Async"))
                >>> print(inspect.iscoroutinefunction(tool.real_fn))
                True

                ```
        """
        if self._real_fn is None:
            raise ValueError("Real function is not set!")

        return self._real_fn

    @staticmethod
    def _parse_tool_output(raw_output: Any) -> List[ChunkType]:
        """Normalize raw function output into a list of content chunks.

        The following conversions are performed:
        - A single :class:`TextChunk`, :class:`Image`, or :class:`Audio` becomes
          a single-item list.
        - A list composed entirely of those chunk types is returned as-is.
        - Any other value is coerced to a :class:`TextChunk` via ``str(value)``.

        Args:
            raw_output (Any): The raw value returned by the wrapped callable.

        Returns:
            List[ChunkType]: A list of content chunks suitable for
            :class:`~serapeum.core.tools.models.ToolOutput`.

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
                >>> from serapeum.core.base.llms.models import TextChunk
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
                >>> from serapeum.core.tools.models import ToolMetadata
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
                >>> from serapeum.core.tools.models import ToolMetadata
                >>> def greet(name: str, punct: str = "!") -> str:
                ...     return f"Hello, {name}{punct}"
                >>> tool = CallableTool(func=greet, metadata=ToolMetadata(name="greet", description="Greet"), default_arguments={"punct": "!!"})
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
                >>> from serapeum.core.tools.models import ToolMetadata
                >>> def concat(a: str, b: str) -> str:
                ...     return a + b
                >>> tool = CallableTool(func=concat, metadata=ToolMetadata(name="concat", description="Concat"))
                >>> print(tool.call("a", "b").content)
                ab

                ```
            - Overriding defaults
                ```python
                >>> from serapeum.core.tools.callable_tool import CallableTool
                >>> from serapeum.core.tools.models import ToolMetadata
                >>> def greet(name: str, punct: str = "!") -> str:
                ...     return f"Hello, {name}{punct}"
                >>> tool = CallableTool(func=greet, metadata=ToolMetadata(name="greet", description="Greet"), default_arguments={"punct": "?"})
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
                >>> from serapeum.core.tools.models import ToolMetadata
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
