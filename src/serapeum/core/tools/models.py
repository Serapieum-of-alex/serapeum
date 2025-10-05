import asyncio
import json
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type
from serapeum.core.base.llms.models import ChunkType, TextChunk
from pydantic import BaseModel


class MinimalToolSchema(BaseModel):
    """Default function-arguments schema for tools.

    This Pydantic model represents the minimal JSON-serializable input shape used by
    tools when no custom schema is provided. It contains a single field, ``input``,
    which is a free-form string. The JSON Schema generated from this model is used
    when constructing function-calling tool specifications.

    Args:
        input (str): The free-form textual input to a tool.

    Returns:
        MinimalToolSchema: An instance validated by Pydantic when constructed.

    Raises:
        pydantic.ValidationError: If required fields are missing or have invalid
            types during instantiation.

    Examples:
        - Typical usage: create and access the default schema instance

            ```python
            >>> from serapeum.core.tools.models import MinimalToolSchema
            >>> schema = MinimalToolSchema(input="what's the weather today?")
            >>> print(schema.input)
            what's the weather today?

            ```
        - Validation error on missing required field
            ```python
            >>> from pydantic import ValidationError
            >>> def build_invalid():
            ...     # missing required field "input"
            ...     MinimalToolSchema()  # type: ignore[call-arg]
            ...
            >>> try:
            ...     build_invalid()
            ... except ValidationError as e:
            ...     print(type(e).__name__)
            ValidationError

            ```

    See Also:
        - ToolMetadata: Holds metadata for a tool, including a custom ``fn_schema``.
        - ToolOutput: Represents the standardized output of a tool.
    """

    input: str


@dataclass
class ToolMetadata:
    """Metadata describing a callable tool and its function-call schema.

    This dataclass encapsulates the name, description, and parameter schema for a
    tool. It can generate an OpenAI-style function tool spec and provide the JSON
    Schema of the function parameters. When no custom schema is supplied, it falls
    back to :class:`MinimalToolSchema`.

    Args:
        description (str):
            A concise, human-readable description of the tool's
            purpose and behavior. Used when exporting the tool to function-calling
            providers. Keep under 1024 characters unless you disable length checks.
        name (Optional[str]):
            The public function name for the tool. This must be
            a valid identifier when exporting to function-calling providers.
        fn_schema (Optional[Type[pydantic.BaseModel]]):
            A Pydantic model class describing the function's input parameters. If ``None``, a simple
            ``{"input": str}`` schema is used.
        return_direct (bool): If ``True``, indicates that the tool's result should
            be returned to the user immediately, bypassing additional LLM steps.

    Returns:
        ToolMetadata: The initialized metadata object.

    Raises:
        None

    Examples:
        - Typical usage with default parameter schema
            ```python
            >>> from serapeum.core.tools.models import ToolMetadata
            >>> meta = ToolMetadata(description="Echo user input back.", name="echo")
            >>> params = meta.get_parameters_dict()
            >>> sorted(set(params) & {"type", "properties", "required"})
            ['properties', 'required', 'type']

            ```

        - Using a custom Pydantic schema for parameters
            ```python
            >>> from pydantic import BaseModel
            >>> class MyArgs(BaseModel):
            ...     query: str
            ...     limit: int | None = None
            ...
            >>> meta = ToolMetadata(description="Search items.", name="search", fn_schema=MyArgs)
            >>> tool_spec = meta.to_openai_tool()
            >>> tool_spec["type"], list(tool_spec["function"])[:2]
            ('function', ['name', 'description'])

            ```

    See Also:
        - MinimalToolSchema: The built-in fallback schema.
        - adapt_to_async_tool: Helper to make a sync tool usable in async flows.
    """

    description: str
    name: Optional[str] = None
    fn_schema: Optional[Type[BaseModel]] = MinimalToolSchema
    return_direct: bool = False

    def get_parameters_dict(self) -> dict:
        """Return the JSON Schema dictionary for this tool's parameters.

        If ``fn_schema`` is ``None``, a minimal schema with a single string field
        named ``input`` is returned. Otherwise, this method derives the schema from
        the provided Pydantic model class and filters it down to the keys relevant
        for function-calling providers.

        Args:
            self (ToolMetadata): The current metadata instance.

        Returns:
            dict: A JSON Schema-like dictionary containing keys such as
                ``type``, ``properties``, ``required``, and optional ``definitions``
                or ``$defs`` depending on the Pydantic version.

        Raises:
            None

        Examples:
            - Default schema when no custom ``fn_schema`` is supplied
                ```python
                >>> from serapeum.core.tools.models import ToolMetadata
                >>> meta = ToolMetadata(description="Echo", name="echo", fn_schema=None)
                >>> params = meta.get_parameters_dict()
                >>> sorted(set(params) & {"type", "properties", "required"})
                ['properties', 'required', 'type']

                ```

            - Schema derived from a custom Pydantic model
                ```python
                >>> from pydantic import BaseModel
                >>> class MyArgs(BaseModel):
                ...     text: str
                ...     count: int
                ...
                >>> meta = ToolMetadata(description="Repeat text", name="repeat", fn_schema=MyArgs)
                >>> params = meta.get_parameters_dict()
                >>> 'properties' in params and set(params['properties']).issuperset({'text', 'count'})
                True

                ```
        """
        if self.fn_schema is None:
            parameters = {
                "type": "object",
                "properties": {
                    "input": {"title": "input query string", "type": "string"},
                },
                "required": ["input"],
            }
        else:
            parameters = self.fn_schema.model_json_schema()
            parameters = {
                k: v
                for k, v in parameters.items()
                if k in ["type", "properties", "required", "definitions", "$defs"]
            }
        return parameters

    @property
    def fn_schema_str(self) -> str:
        """Return the function-argument schema as a JSON string.

        This property serializes the parameter schema produced by
        :meth:`get_parameters_dict` to a JSON string. If ``fn_schema`` is ``None``,
        it raises a ``ValueError`` to make the absence explicit (use
        :meth:`get_parameters_dict` directly to obtain the default schema).

        Args:
            self (ToolMetadata): The current metadata instance.

        Returns:
            str: A JSON string encoding of the parameters schema.

        Raises:
            ValueError: If ``fn_schema`` is ``None``. In that case, call
                :meth:`get_parameters_dict` instead to retrieve the default schema.

        Examples:
            - Successful serialization of a custom schema
                ```python
                >>> import json
                >>> from pydantic import BaseModel
                >>> from serapeum.core.tools.models import ToolMetadata
                >>> class Args(BaseModel):
                ...     input: str
                ...
                >>> meta = ToolMetadata(description="Echo", name="echo", fn_schema=Args)
                >>> s = meta.fn_schema_str
                >>> isinstance(json.loads(s), dict)
                True

                ```

            - Error when ``fn_schema`` is ``None``
                ```python
                >>> from serapeum.core.tools.models import ToolMetadata
                >>> meta = ToolMetadata(description="Echo", name="echo", fn_schema=None)
                >>> try:
                ...     _ = meta.fn_schema_str
                ... except ValueError as e:
                ...     print(str(e))
                fn_schema is None.

                ```
        """
        if self.fn_schema is None:
            raise ValueError("fn_schema is None.")
        parameters = self.get_parameters_dict()
        return json.dumps(parameters, ensure_ascii=False)

    def get_name(self) -> str:
        """Return the tool's declared name.

        This helper ensures the name is present and raises an error if it is not,
        which is useful before exporting a tool to function-calling providers.

        Args:
            self (ToolMetadata): The current metadata instance.

        Returns:
            str: The non-empty tool name.

        Raises:
            ValueError: If ``name`` is ``None``.

        Examples:
            - Retrieve a valid name
                ```python
                >>> from serapeum.core.tools.models import ToolMetadata
                >>> meta = ToolMetadata(description="Echo", name="echo")
                >>> meta.get_name()
                'echo'

                ```

            - Error when name is missing
                ```python
                >>> from serapeum.core.tools.models import ToolMetadata
                >>> meta = ToolMetadata(description="No name provided", name=None)
                >>> try:
                ...     meta.get_name()
                ... except ValueError as e:
                ...     print(str(e))
                name is None.

                ```
        """
        if self.name is None:
            raise ValueError("name is None.")
        return self.name

    def to_openai_tool(self, skip_length_check: bool = False) -> Dict[str, Any]:
        """Export this metadata as an OpenAI function-calling tool spec.

        Builds a dictionary compatible with OpenAI-style function tools. By default,
        this enforces a 1024-character limit on ``description`` to match common
        provider constraints.

        Args:
            skip_length_check (bool): If ``True``, bypass validation of the
                description length. Defaults to ``False``.

        Returns:
            Dict[str, Any]: A dictionary with keys ``type`` and ``function``. The
                latter contains ``name``, ``description``, and ``parameters``.

        Raises:
            ValueError: If ``description`` exceeds 1024 characters and
                ``skip_length_check`` is ``False``.

        Examples:
            - Export with default schema
                ```python
                >>> from serapeum.core.tools.models import ToolMetadata
                >>> meta = ToolMetadata(description="Echo input.", name="echo")
                >>> spec = meta.to_openai_tool()
                >>> spec["type"], sorted(spec["function"])  # doctest: +NORMALIZE_WHITESPACE
                ('function', ['description', 'name', 'parameters'])

                ```

            - Error on overly long descriptions (when not skipped)
                ```python
                >>> from serapeum.core.tools.models import ToolMetadata
                >>> long_desc = "x" * 1025
                >>> meta = ToolMetadata(description=long_desc, name="tool")
                >>> try:
                ...     meta.to_openai_tool()
                ... except ValueError as e:
                ...     print("exceeds" in str(e))
                True

                ```
        """
        if not skip_length_check and len(self.description) > 1024:
            raise ValueError(
                "Tool description exceeds maximum length of 1024 characters. "
                "Please shorten your description or move it to the prompt."
            )
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters_dict(),
            },
        }


class ToolOutput(BaseModel):
    """Standardized output container returned by tools.

    Tool outputs may include structured chunks (e.g., text, images, audio) as well
    as raw input/output payloads from the underlying tool. The ``content`` helper
    provides a convenient view over text chunks (instances of
    :class:`serapeum.core.base.llms.models.TextChunk`).

    Args:
        chunks (List[ChunkType]): A list of content chunks. If ``content`` is
            supplied, this value is ignored and replaced by a single ``TextChunk``.
        tool_name (str): The name of the tool that produced this output.
        raw_input (Dict[str, Any]): Implementation-specific input payload for
            debugging or provenance.
        raw_output (Any): Implementation-specific raw output, if any.
        is_error (bool): Whether this output represents an error condition.

    Returns:
        ToolOutput: A validated Pydantic model instance.

    Raises:
        ValueError: If both ``content`` and ``chunks`` are provided in the
            constructor (see ``__init__``).

    Examples:
        - Typical usage with string content

            ```python
            >>> from serapeum.core.tools.models import ToolOutput
            >>> out = ToolOutput(tool_name="echo", content="hello")
            >>> out.content
            'hello'

            ```

        - Using chunks directly
            ```python
            >>> from serapeum.core.base.llms.models import TextChunk
            >>> from serapeum.core.tools.models import ToolOutput
            >>> out = ToolOutput(tool_name="echo", chunks=[TextChunk(content="hi")])
            >>> isinstance(out.chunks[0], TextChunk) and out.content
            'hi'

            ```

        - Edge case: providing both ``content`` and ``chunks`` raises an error

            ```python
            >>> from serapeum.core.base.llms.models import TextChunk
            >>> from serapeum.core.tools.models import ToolOutput
            >>> try:
            ...     ToolOutput(tool_name="echo", content="x", chunks=[TextChunk(content="y")])
            ... except ValueError as e:
            ...     print("Cannot provide both" in str(e))
            True

            ```

    See Also:
        - MinimalToolSchema: Default input schema for tools.
        - ToolMetadata: Describes the tool that produced this output.
    """

    chunks: List[ChunkType]
    tool_name: str
    raw_input: Dict[str, Any]
    raw_output: Any
    is_error: bool = False

    def __init__(
        self,
        tool_name: str,
        content: Optional[str] = None,
        chunks: Optional[List[ChunkType]] = None,
        raw_input: Optional[Dict[str, Any]] = None,
        raw_output: Optional[Any] = None,
        is_error: bool = False,
    ):
        """Initialize a ToolOutput instance.

        Exactly one of ``content`` or ``chunks`` may be provided. If ``content`` is
        given, it is wrapped into a single :class:`~serapeum.core.base.llms.models.TextChunk`.

        Args:
            tool_name (str): The name of the producing tool.
            content (Optional[str]): A convenience text payload. If supplied,
                ``chunks`` must be omitted.
            chunks (Optional[List[ChunkType]]): Explicit chunk list. If supplied,
                ``content`` must be omitted.
            raw_input (Optional[Dict[str, Any]]): Optional debug/provenance input.
            raw_output (Optional[Any]): Optional raw output from the tool.
            is_error (bool): Flag indicating the output is an error.

        Returns:
            None: The constructed instance.

        Raises:
            ValueError: If both ``content`` and ``chunks`` are provided.

        Examples:
            - Construct with content
                ```python
                >>> from serapeum.core.tools.models import ToolOutput
                >>> out = ToolOutput(tool_name="echo", content="hello")
                >>> out.content
                'hello'

                ```
            - Construct with chunks
                ```python
                >>> from serapeum.core.base.llms.models import TextChunk
                >>> from serapeum.core.tools.models import ToolOutput
                >>> out = ToolOutput(tool_name="echo", chunks=[TextChunk(content="hello")])
                >>> out.content
                'hello'

                ```
        """
        if content and chunks:
            raise ValueError("Cannot provide both content and chunks.")
        if content:
            chunks = [TextChunk(content=content)]
        elif chunks:
            pass
        else:
            chunks = []

        super().__init__(
            tool_name=tool_name,
            chunks=chunks,
            raw_input=raw_input,
            raw_output=raw_output,
            is_error=is_error,
        )

    @property
    def content(self) -> str:
        """Return a unified text view over all text chunks.

        Aggregates the textual content of all :class:`TextChunk` instances present in
        ``self.chunks``, joined by newlines. Non-text chunks are ignored.

        Args:
            self (ToolOutput): The current tool output instance.

        Returns:
            str: The concatenated textual content, or an empty string if there are
                no text chunks.

        Raises:
            None

        Examples:
            - Multiple text chunks are joined with a newline

                ```python
                >>> from serapeum.core.base.llms.models import TextChunk
                >>> from serapeum.core.tools.models import ToolOutput
                >>> out = ToolOutput(tool_name="t", chunks=[TextChunk(content="a"), TextChunk(content="b")])
                >>> out.content
                'a\nb'

                ```
        """
        return "\n".join(
            [chunk.content for chunk in self.chunks if isinstance(chunk, TextChunk)]
        )

    @content.setter
    def content(self, content: str) -> None:
        """Overwrite the text content with a single TextChunk.

        This setter replaces any existing chunks with a single
        :class:`~serapeum.core.base.llms.models.TextChunk` containing ``content``.

        Args:
            content (str): The new text content to set.

        Returns:
            None

        Raises:
            None

        Examples:
            - Overwrite existing chunks
                ```python
                >>> from serapeum.core.tools.models import ToolOutput
                >>> out = ToolOutput(tool_name="t", content="first")
                >>> out.content
                'first'
                >>> out.content = "second"
                >>> out.content
                'second'

                ```
        """
        self.chunks = [TextChunk(content=content)]

    def __str__(self) -> str:
        """Return the string representation of this output.

        The string form is equivalent to ``self.content`` and therefore contains
        the concatenated text of all text chunks.

        Args:
            self (ToolOutput): The current instance.

        Returns:
            str: The same value as :pyattr:`ToolOutput.content`.

        Raises:
            None

        Examples:
            - String conversion mirrors ``content``

                ```python

                >>> from serapeum.core.tools.models import ToolOutput
                >>> out = ToolOutput(tool_name="t", content="hello")
                >>> str(out)
                'hello'

                ```
        """
        return self.content


class BaseTool:
    """Synchronous tool interface.

    Implementations should provide :pyattr:`metadata` and implement ``__call__``
    to perform the tool's logic. For async compatibility, see
    :class:`AsyncBaseTool` and :func:`adapt_to_async_tool`.

    Examples:
        - Minimal echo tool

            ```python

            >>> from serapeum.core.tools.models import BaseTool, ToolMetadata, ToolOutput
            >>> class Echo(BaseTool):
            ...     @property
            ...     def metadata(self) -> ToolMetadata:
            ...         return ToolMetadata(description="Echo input.", name="echo")
            ...
            ...     def __call__(self, input_values: dict) -> ToolOutput:
            ...         text = input_values.get("input", "")
            ...         return ToolOutput(tool_name=self.metadata.get_name(), content=text)
            ...
            >>> tool = Echo()
            >>> out = tool({"input": "hi"})
            >>> out.content
            'hi'

            ```
    """

    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        """Metadata describing this tool.

        Returns:
            ToolMetadata: The tool's metadata including name, description, and
                input schema.
        """
        pass

    @abstractmethod
    def __call__(self, input_values: Any) -> ToolOutput:
        """Execute the tool synchronously.

        Args:
            input_values (Any): The inputs for the tool. Strongly recommended to be
                a dict matching the tool's ``fn_schema``.

        Returns:
            ToolOutput: The structured tool output.
        """
        pass


class AsyncBaseTool(BaseTool):
    """Async-capable tool interface.

    Subclasses should implement both :meth:`call` (sync) and :meth:`acall` (async)
    to support a wide range of execution contexts. ``__call__`` delegates to the
    synchronous :meth:`call` by default. Use :func:`adapt_to_async_tool` to adapt a
    purely synchronous :class:`BaseTool` into an async-capable tool.

    Examples:
        - Minimal async echo tool

            ```python

            >>> import asyncio
            >>> from serapeum.core.tools.models import AsyncBaseTool, ToolMetadata, ToolOutput
            >>> class EchoAsync(AsyncBaseTool):
            ...     @property
            ...     def metadata(self) -> ToolMetadata:
            ...         return ToolMetadata(description="Echo input (async).", name="echo_async")
            ...
            ...     def call(self, input_values: dict) -> ToolOutput:
            ...         text = input_values.get("input", "")
            ...         return ToolOutput(tool_name=self.metadata.get_name(), content=text)
            ...
            ...     async def acall(self, input_values: dict) -> ToolOutput:
            ...         await asyncio.sleep(0)  # simulate async work
            ...         return self.call(input_values)
            ...
            >>> tool = EchoAsync()
            >>> asyncio.run(tool.acall({"input": "hello"})).content
            'hello'

            ```
    """

    def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Invoke the synchronous :meth:`call` implementation.

        This convenience allows an :class:`AsyncBaseTool` to be used in synchronous
        contexts transparently.

        Args:
            *args: Positional arguments forwarded to :meth:`call`.
            **kwargs: Keyword arguments forwarded to :meth:`call`.

        Returns:
            ToolOutput: The result of the synchronous call.
        """
        return self.call(*args, **kwargs)

    @abstractmethod
    def call(self, input_values: Any) -> ToolOutput:
        """Synchronous implementation of the tool.

        Args:
            input_values (Any): Input values, typically a dict matching the
                tool's input schema.

        Returns:
            ToolOutput: The tool's output.
        """

    @abstractmethod
    async def acall(self, input_values: Any) -> ToolOutput:
        """Asynchronous implementation of the tool.

        Args:
            input_values (Any): Input values, typically a dict matching the
                tool's input schema.

        Returns:
            ToolOutput: The tool's output resolved asynchronously.
        """


class BaseToolAsyncAdapter(AsyncBaseTool):
    """Adapter that exposes a synchronous :class:`BaseTool` as async-compatible.

    It forwards synchronous calls to the wrapped tool and implements
    :meth:`acall` by running the sync call in a worker thread via
    :func:`asyncio.to_thread`.

    Examples:
        - Adapt a synchronous tool to async usage

            ```python

            >>> import asyncio
            >>> from serapeum.core.tools.models import BaseTool, ToolMetadata, ToolOutput, BaseToolAsyncAdapter
            >>> class Echo(BaseTool):
            ...     @property
            ...     def metadata(self) -> ToolMetadata:
            ...         return ToolMetadata(description="Echo", name="echo")
            ...
            ...     def __call__(self, input_values: dict) -> ToolOutput:
            ...         return ToolOutput(tool_name="echo", content=input_values.get("input", ""))
            ...
            >>> adapter = BaseToolAsyncAdapter(Echo())
            >>> asyncio.run(adapter.acall({"input": "hi"})).content
            'hi'

            ```
    """

    def __init__(self, tool: BaseTool):
        """Create a new adapter for a synchronous tool.

        Args:
            tool (BaseTool): The synchronous tool instance to adapt.

        Returns:
            None
        """
        self.base_tool = tool

    @property
    def metadata(self) -> ToolMetadata:
        """Proxy the underlying tool's metadata.

        Returns:
            ToolMetadata: The wrapped tool's metadata.
        """
        return self.base_tool.metadata

    def call(self, input_values: Any) -> ToolOutput:
        """Synchronously forward to the wrapped tool.

        Args:
            input_values (Any): Inputs for the underlying tool.

        Returns:
            ToolOutput: The result from the underlying tool.
        """
        return self.base_tool(input_values)

    async def acall(self, input_values: Any) -> ToolOutput:
        """Run the sync call in a thread to provide async semantics.

        Args:
            input_values (Any): Inputs for the underlying tool.

        Returns:
            ToolOutput: The result, awaited from a worker thread.
        """
        return await asyncio.to_thread(self.call, input_values)


def adapt_to_async_tool(tool: BaseTool) -> AsyncBaseTool:
    """Return an async-capable tool, adapting sync tools when necessary.

    If ``tool`` already subclasses :class:`AsyncBaseTool`, it is returned as-is.
    Otherwise, it is wrapped with :class:`BaseToolAsyncAdapter` to provide an
    asynchronous interface via :meth:`AsyncBaseTool.acall`.

    Args:
        tool (BaseTool): The tool instance to make async-capable.

    Returns:
        AsyncBaseTool: Either the original tool (if already async-capable) or an
            adapter around it.

    Raises:
        None

    Examples:
        - Passing through an already-async tool

            ```python

            >>> import asyncio
            >>> from serapeum.core.tools.models import AsyncBaseTool, ToolMetadata, ToolOutput, adapt_to_async_tool
            >>> class EchoAsync(AsyncBaseTool):
            ...     @property
            ...     def metadata(self) -> ToolMetadata:
            ...         return ToolMetadata(description="Echo (async)", name="echo_async")
            ...
            ...     def call(self, input_values: dict) -> ToolOutput:
            ...         return ToolOutput(tool_name="echo_async", content=input_values.get("input", ""))
            ...
            ...     async def acall(self, input_values: dict) -> ToolOutput:
            ...         return self.call(input_values)
            ...
            >>> async_tool = adapt_to_async_tool(EchoAsync())
            >>> asyncio.run(async_tool.acall({"input": "ok"})).content
            'ok'

            ```

        - Adapting a synchronous tool

            ```python

            >>> import asyncio
            >>> from serapeum.core.tools.models import BaseTool, ToolMetadata, ToolOutput, adapt_to_async_tool
            >>> class Echo(BaseTool):
            ...     @property
            ...     def metadata(self) -> ToolMetadata:
            ...         return ToolMetadata(description="Echo", name="echo")
            ...
            ...     def __call__(self, input_values: dict) -> ToolOutput:
            ...         return ToolOutput(tool_name="echo", content=input_values.get("input", ""))
            ...
            >>> async_tool = adapt_to_async_tool(Echo())
            >>> asyncio.run(async_tool.acall({"input": "hi"})).content
            'hi'

            ```
    """
    if isinstance(tool, AsyncBaseTool):
        return tool
    else:
        return BaseToolAsyncAdapter(tool)
