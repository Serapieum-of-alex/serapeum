import asyncio
import inspect
from typing import Any, Awaitable, Callable, Optional, Type, Dict, Union, List, Tuple
import re
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
from serapeum.core.tools.utils import create_schema_from_function

AsyncCallable = Callable[..., Awaitable[Any]]



def sync_to_async(fn: Callable[..., Any]) -> AsyncCallable:
    """Sync to async."""

    async def _async_wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    return _async_wrapped_fn


def async_to_sync(func_async: AsyncCallable) -> Callable:
    """Async from sync."""

    def _sync_wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        return asyncio_run(func_async(*args, **kwargs))  # type: ignore[arg-type]

    return _sync_wrapped_fn


class FunctionTool(AsyncBaseTool):
    """Function Tool.

    A tool that takes in a function and optionally handles workflow context.
    """

    def __init__(
        self,
        fn: Optional[Callable[..., Any]] = None,
        metadata: Optional[ToolMetadata] = None,
        async_fn: Optional[AsyncCallable] = None,
        partial_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if fn is None and async_fn is None:
            raise ValueError("fn or async_fn must be provided.")

        # Handle function (sync and async)
        self._real_fn = fn or async_fn
        if async_fn is not None:
            self._async_fn = async_fn
            self._fn = fn or async_to_sync(async_fn)
        else:
            assert fn is not None
            if inspect.iscoroutinefunction(fn):
                self._async_fn = fn
                self._fn = async_to_sync(fn)
            else:
                self._fn = fn
                self._async_fn = sync_to_async(fn)

        # Determine if the function requires context by inspecting its signature
        fn_to_inspect = fn or async_fn
        assert fn_to_inspect is not None
        # sig = inspect.signature(fn_to_inspect)
        self.requires_context = False

        if metadata is None:
            raise ValueError("metadata must be provided")

        self._metadata = metadata
        self.partial_params = partial_params or {}

    @classmethod
    def from_defaults(
        cls,
        fn: Optional[Callable[..., Any]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        tool_schema: Optional[Type[BaseModel]] = None,
        async_fn: Optional[AsyncCallable] = None,
        tool_metadata: Optional[ToolMetadata] = None,
        partial_params: Optional[Dict[str, Any]] = None,
    ) -> "FunctionTool":
        partial_params = partial_params or {}

        if tool_metadata is None:
            fn_to_parse = fn or async_fn
            assert fn_to_parse is not None, "fn must be provided"
            name = name or fn_to_parse.__name__
            docstring = fn_to_parse.__doc__ or ""

            # Get function signature
            fn_sig = inspect.signature(fn_to_parse)
            fn_params = set(fn_sig.parameters.keys())

            # 1. Extract docstring param descriptions
            param_docs, unknown_params = cls.extract_param_docs(docstring, fn_params)

            # Handle FieldInfo defaults (remove default values and make all parameters required)
            fn_sig = fn_sig.replace(
                parameters=[
                    param.replace(default=inspect.Parameter.empty)
                    if isinstance(param.default, FieldInfo)
                    else param
                    for param in fn_sig.parameters.values()
                ]
            )

            # 5. Build enriched description using param_docs
            if description is None:
                description = f"{name}{fn_sig}\n"

                # Extract the first meaningful line (summary) from the docstring
                doc_lines = docstring.strip().splitlines()
                for line in doc_lines:
                    if line.strip():
                        description += line.strip()
                        break


            # 6. Build tool_schema only if not already provided
            if tool_schema is None:

                tool_schema = create_schema_from_function(
                    f"{name}",
                    fn_to_parse,
                    additional_fields=None,
                    ignore_fields=None,
                )
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
            fn=fn,
            metadata=tool_metadata,
            async_fn=async_fn,
            partial_params=partial_params,
        )

    @staticmethod
    def extract_param_docs(
            docstring: str, fn_params: Optional[set] = None
    ) -> Tuple[dict, set]:
        """
        Parses param descriptions from a docstring.

        Returns:
            - param_docs: Only for params in fn_params with non-conflicting descriptions.
            - unknown_params: Params found in docstring but not in fn_params (ignored in final output).

        """
        raw_param_docs: dict[str, str] = {}
        unknown_params = set()

        def try_add_param(name: str, desc: str) -> None:
            desc = desc.strip()
            if fn_params and name not in fn_params:
                unknown_params.add(name)
                return
            if name in raw_param_docs and raw_param_docs[name] != desc:
                return
            raw_param_docs[name] = desc

        # Sphinx style
        for match in re.finditer(r":param (\w+): (.+)", docstring):
            try_add_param(match.group(1), match.group(2))

        # Google style
        for match in re.finditer(
                r"^\s*(\w+)\s*\(.*?\):\s*(.+)$", docstring, re.MULTILINE
        ):
            try_add_param(match.group(1), match.group(2))

        # Javadoc style
        for match in re.finditer(r"@param (\w+)\s+(.+)", docstring):
            try_add_param(match.group(1), match.group(2))

        return raw_param_docs, unknown_params

    @property
    def metadata(self) -> ToolMetadata:
        """Metadata."""
        return self._metadata

    @property
    def fn(self) -> Callable[..., Any]:
        """Function."""
        return self._fn

    @property
    def async_fn(self) -> AsyncCallable:
        """Async function."""
        return self._async_fn

    @property
    def real_fn(self) -> Union[Callable[..., Any], AsyncCallable]:
        """Real function."""
        if self._real_fn is None:
            raise ValueError("Real function is not set!")

        return self._real_fn

    def _parse_tool_output(self, raw_output: Any) -> List[ChunkType]:
        """Parse tool output into content chunks."""
        if isinstance(raw_output, (TextChunk, Image, Audio)):
            return [raw_output]
        elif isinstance(raw_output, list) and all(isinstance(item, (TextChunk, Image, Audio)) for item in raw_output):
            return raw_output
        else:
            return [TextChunk(content=str(raw_output))]

    def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
        all_kwargs = {**self.partial_params, **kwargs}
        return self.call(*args, **all_kwargs)

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Sync Call."""
        all_kwargs = {**self.partial_params, **kwargs}

        raw_output = self._fn(*args, **all_kwargs)

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
        """Async Call."""
        all_kwargs = {**self.partial_params, **kwargs}

        raw_output = await self._async_fn(*args, **all_kwargs)

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
