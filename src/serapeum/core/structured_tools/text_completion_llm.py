from typing import Any, Dict, Optional, Type, Union, Tuple

from pydantic import BaseModel
from serapeum.core.llm.base import LLM
from serapeum.core.output_parsers.models import PydanticOutputParser
from serapeum.core.prompts.base import BasePromptTemplate, PromptTemplate
from serapeum.core.configs.configs import Configs
from serapeum.core.structured_tools.models import BasePydanticLLM
from serapeum.core.output_parsers import BaseOutputParser


class TextCompletionLLM(BasePydanticLLM[BaseModel]):
    """Structured text completion runner that returns typed Pydantic models.

    The wrapper binds a prompt, an output parser, and an LLM together so every invocation returns a
    validated Pydantic model that matches the declared schema.

    Args:
        output_parser (Optional[BaseOutputParser]): Parser used to coerce raw text into the target
            model. Required when `output_cls` is not supplied.
        prompt (Union[BasePromptTemplate, str]): Prompt template or raw template string that drives
            the LLM request.
        output_cls (Optional[Type[BaseModel]]): Pydantic model that constrains the expected
            structure. When omitted, it is inferred from a `PydanticOutputParser`.
        llm (Optional[LLM]): Concrete language model implementation. Falls back to `Configs.llm`
            when left unset.
        verbose (bool): Enables verbose tracing in the underlying base class.

    Returns:
        TextCompletionLLM: Fully configured structured completion program.

    Raises:
        ValueError: If the prompt cannot be converted to a `BasePromptTemplate` or the parser
            configuration is invalid.
        AssertionError: If no LLM instance is provided and `Configs.llm` is not configured.

    Examples:
        - Basic synchronous usage with a string prompt
            ```python
            >>> from types import SimpleNamespace
            >>> from pydantic import BaseModel
            >>> from serapeum.core.output_parsers.models import PydanticOutputParser
            >>> from serapeum.core.structured_tools.text_completion_llm import TextCompletionLLM
            >>> from serapeum.llms.ollama import Ollama
            >>> LLM = Ollama(
            ...     model="llama3.1",
            ...     request_timeout=180,
            >>> )
            >>> class Greeting(BaseModel):
            ...     message: str
            >>>
            >>> tool = TextCompletionLLM(
            ...     output_parser=PydanticOutputParser(output_cls=Greeting),
            ...     prompt="message",
            ...     llm=LLM,
            ... )
            >>> tool() # doctest: +SKIP
            Greeting(message='Hello, World')

            ```
        - Chat-model usage with templated prompts
            ```python
            >>> from types import SimpleNamespace
            >>> from pydantic import BaseModel
            >>> from serapeum.core.prompts.base import PromptTemplate
            >>> from serapeum.core.output_parsers.models import PydanticOutputParser
            >>> class Greeting(BaseModel):
            ...     message: str
            >>> prompt = PromptTemplate("Say hello to {name}.")
            >>> tool = TextCompletionLLM(
            ...     output_parser=PydanticOutputParser(output_cls=Greeting),
            ...     prompt=prompt,
            ...     llm=LLM,
            ... )
            >>> tool(name="Bob") # doctest: +SKIP
            Greeting(message='hi')

            ```

    See Also:
        TextCompletionLLM.__call__: Synchronous inference entry point.
        TextCompletionLLM.acall: Asynchronous inference entry point.
    """

    def __init__(
        self,
        *,
        output_parser: Optional[BaseOutputParser] = None,
        prompt: Union[BasePromptTemplate, str],
        output_cls: Optional[Type[BaseModel]] = None,
        llm: Optional[LLM] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the structured completion pipeline.

        Args:
            output_parser (Optional[BaseOutputParser]): Parser responsible for translating the raw
                LLM response into a structured object.
            prompt (Union[BasePromptTemplate, str]): Prompt template or string used to query the
                LLM.
            output_cls (Optional[Type[BaseModel]]): Target Pydantic model type. When omitted, a
                compatible `PydanticOutputParser` must be supplied.
            llm (Optional[LLM]): Language model implementation backing the prompt execution.
            verbose (bool): Enables verbose tracing inherited from the base class.

        Returns:
            None: This initializer mutates the instance in place.

        Raises:
            ValueError: Raised when the prompt or output parser cannot be validated.
            AssertionError: Raised when neither an LLM argument nor `Configs.llm` is set.

        Examples:
            - Auto-create a parser from an output class
                ```python
                >>> from types import SimpleNamespace
                >>> from pydantic import BaseModel
                >>> from serapeum.llms.ollama import Ollama
                >>> LLM = Ollama(
                ...     model="llama3.1",
                ...     request_timeout=180,
                >>> )
                >>> class Item(BaseModel):
                ...     name: str
                >>> text_llm = TextCompletionLLM(
                ...     output_cls=Item,
                ...     prompt="Name a thing",
                ...     llm=LLM,
                ... )
                >>> text_llm.output_cls is Item
                True

                ```
            - Surface missing-LLM misconfiguration
                ```python
                >>> from pydantic import BaseModel
                >>> from serapeum.core.configs.configs import Configs
                >>> class Item(BaseModel):
                ...     name: str
                >>> Configs.llm = None
                >>> TextCompletionLLM(output_cls=Item, prompt="Name a thing", llm=None)
                Traceback (most recent call last):
                ...
                AssertionError: llm must be provided or set in Configs.

                ```

        See Also:
            TextCompletionLLM.validate_output_parser_cls: Validates parser and output class pairing.
            TextCompletionLLM.validate_llm: Resolves the backing LLM instance.
        """
        self._output_parser, self._output_cls = self.validate_output_parser_cls(output_parser, output_cls)
        self._llm = self.validate_llm(llm)
        self._prompt = self.validate_prompt(prompt)
        self._verbose = verbose
        self._prompt.output_parser = output_parser

    @staticmethod
    def validate_prompt(prompt: Union[BasePromptTemplate, str]) -> BasePromptTemplate:
        """Validate that the provided prompt is usable by the program.

        Args:
            prompt (Union[BasePromptTemplate, str]): Candidate prompt configuration that should be
                a template instance or raw string.

        Returns:
            BasePromptTemplate: Normalized prompt template ready for rendering.

        Raises:
            ValueError: Raised when `prompt` is neither a string nor a `BasePromptTemplate`.

        Examples:
            - Promote a raw string into a prompt template
                ```python
                >>> from serapeum.core.structured_tools.text_completion_llm import TextCompletionLLM
                >>> validated = TextCompletionLLM.validate_prompt("Hello {name}")
                >>> validated.get_template()
                'Hello {name}'

                ```
            - Reject unsupported prompt types
                ```python
                >>> TextCompletionLLM.validate_prompt(42)
                Traceback (most recent call last):
                ...
                ValueError: prompt must be an instance of BasePromptTemplate or str.

                ```

        See Also:
            serapeum.core.prompts.base.PromptTemplate: Concrete template implementation.
            TextCompletionLLM.__init__: Calls this validator during initialization.
        """
        if not isinstance(prompt, (BasePromptTemplate, str)):
            raise ValueError(
                "prompt must be an instance of BasePromptTemplate or str."
            )
        if isinstance(prompt, str):
            prompt = PromptTemplate(prompt)
        return prompt

    @staticmethod
    def validate_llm(llm: LLM) -> LLM:
        """Resolve the LLM backing structured completions.

        Args:
            llm (Optional[LLM]): Explicit language model to use. When omitted, falls back to the
                global `Configs.llm`.

        Returns:
            LLM: Concrete language model instance that will execute prompts.

        Raises:
            AssertionError: If no LLM is supplied and the global configs do not define one.

        Examples:
            - Return the provided instance without consulting global configs
                ```python
                >>> from types import SimpleNamespace
                >>> from serapeum.core.configs.configs import Configs
                >>> from serapeum.core.structured_tools.text_completion_llm import TextCompletionLLM
                >>> Configs.llm = None
                >>> supplied = SimpleNamespace(metadata=SimpleNamespace(is_chat_model=False))
                >>> TextCompletionLLM.validate_llm(supplied) is supplied
                True

                ```
            - Pull the default instance from `Configs`
                ```python
                >>> from types import SimpleNamespace
                >>> from serapeum.core.configs.configs import Configs
                >>> from serapeum.core.structured_tools.text_completion_llm import TextCompletionLLM
                >>> fallback = SimpleNamespace(metadata=SimpleNamespace(is_chat_model=False))
                >>> Configs.llm = fallback
                >>> TextCompletionLLM.validate_llm(None) is fallback
                True

                ```
            - Raise when no model is available
                ```python
                >>> from serapeum.core.configs.configs import Configs
                >>> from serapeum.core.structured_tools.text_completion_llm import TextCompletionLLM
                >>> Configs.llm = None
                >>> TextCompletionLLM.validate_llm(None)
                Traceback (most recent call last):
                ...
                AssertionError: llm must be provided or set in Configs.

                ```

        See Also:
            serapeum.core.configs.configs.Configs: Houses the global LLM configuration.
        """
        llm = llm or Configs.llm  # type: ignore
        if llm is None:
            raise AssertionError("llm must be provided or set in Configs.")
        return llm

    @staticmethod
    def validate_output_parser_cls(
        output_parser: BaseOutputParser, output_cls: Type[BaseModel]
    ) -> Tuple[BaseOutputParser, Type[BaseModel]]:
        """Validate and normalize parser/schema configuration.

        Args:
            output_parser (Optional[BaseOutputParser]):
                Parser responsible for producing structured responses.
            output_cls (Optional[Type[BaseModel]]):
                Target Pydantic model that defines the schema.

        Returns:
            Tuple[BaseOutputParser, Type[BaseModel]]:
                A parser/schema pair ready for execution.

        Raises:
            ValueError: If neither an `output_cls` nor a compatible `PydanticOutputParser` is
                supplied.

        Examples:
            - Derive the output class from a supplied parser
                ```python
                >>> from pydantic import BaseModel
                >>> from serapeum.core.output_parsers.models import PydanticOutputParser
                >>> from serapeum.core.structured_tools.text_completion_llm import TextCompletionLLM
                >>> class Record(BaseModel):
                ...     value: int
                >>> parser = PydanticOutputParser(output_cls=Record)
                >>> resolved_parser, resolved_cls = TextCompletionLLM.validate_output_parser_cls(
                ...     parser,
                ...     None,  # type: ignore[arg-type]
                ... )
                >>> resolved_parser is parser
                True
                >>> resolved_cls is Record
                True

                ```
            - Auto-create a parser when only the schema is provided
                ```python
                >>> from pydantic import BaseModel
                >>> from serapeum.core.output_parsers.models import PydanticOutputParser
                >>> from serapeum.core.structured_tools.text_completion_llm import TextCompletionLLM
                >>> class Item(BaseModel):
                ...     name: str
                >>> parser, schema = TextCompletionLLM.validate_output_parser_cls(
                ...     None,  # type: ignore[arg-type]
                ...     Item,
                ... )
                >>> isinstance(parser, PydanticOutputParser)
                True
                >>> schema is Item
                True

                ```
            - Reject unsupported parser types without a schema
                ```python
                >>> from serapeum.core.output_parsers import BaseOutputParser
                >>> from serapeum.core.structured_tools.text_completion_llm import TextCompletionLLM
                >>> class PlainParser(BaseOutputParser):
                ...     def parse(self, output: str):
                ...         return output
                >>> TextCompletionLLM.validate_output_parser_cls(
                ...     PlainParser(),
                ...     None,  # type: ignore[arg-type]
                ... )
                Traceback (most recent call last):
                ...
                ValueError: Output parser must be PydanticOutputParser.

                ```

        See Also:
            serapeum.core.output_parsers.models.PydanticOutputParser: Default parser implementation.
        """
        # decide default output class if not set
        if output_cls is None:
            if not isinstance(output_parser, PydanticOutputParser):
                raise ValueError("Output parser must be PydanticOutputParser.")
            output_cls = output_parser.output_cls
        else:
            if output_parser is None:
                output_parser = PydanticOutputParser(output_cls=output_cls)

        return output_parser, output_cls

    @property
    def output_cls(self) -> Type[BaseModel]:
        """Return the Pydantic model produced by this program.

        Returns:
            Type[BaseModel]: Schema guaranteed for every parsed completion.

        Examples:
            - Inspect the configured schema
                ```python
                >>> from types import SimpleNamespace
                >>> from pydantic import BaseModel
                >>> from serapeum.core.output_parsers.models import PydanticOutputParser
                >>> from serapeum.core.structured_tools.text_completion_llm import TextCompletionLLM
                >>> from serapeum.llms.ollama import Ollama
                >>> LLM = Ollama(
                ...     model="llama3.1",
                ...     request_timeout=180,
                >>> )
                >>> class Item(BaseModel):
                ...     value: int
                >>> parser = PydanticOutputParser(output_cls=Item)
                >>> text_llm = TextCompletionLLM(output_parser=parser, prompt="?", llm=LLM)
                >>> text_llm.output_cls is Item
                True

                ```

        See Also:
            TextCompletionLLM.__init__: Details rules used to derive `output_cls`.
        """
        return self._output_cls

    @property
    def prompt(self) -> BasePromptTemplate:
        """Expose the prompt template bound to this program.

        Returns:
            BasePromptTemplate: Prompt used to render requests for the LLM.

        Raises:
            AttributeError:
                Propagated if the stored prompt was overwritten with an incompatible object.

        Examples:
            - Inspect the configured prompt template
                ```python
                >>> from types import SimpleNamespace
                >>> from pydantic import BaseModel
                >>> from serapeum.core.output_parsers.models import PydanticOutputParser
                >>> from serapeum.core.structured_tools.text_completion_llm import TextCompletionLLM
                >>> from serapeum.llms.ollama import Ollama
                >>> LLM = Ollama(
                ...     model="llama3.1",
                ...     request_timeout=180,
                >>> )
                >>> class Item(BaseModel):
                ...     value: int
                >>> text_llm = TextCompletionLLM(
                ...     output_cls=Item,
                ...     prompt="Value?",
                ...     llm=LLM,
                ... )
                >>> text_llm.prompt.get_template()
                'Value?'

                ```

        See Also:
            TextCompletionLLM.__init__: Establishes the initial prompt value.
            TextCompletionLLM.prompt.fset: Setter that updates the stored prompt.
        """
        return self._prompt

    @prompt.setter
    def prompt(self, prompt: BasePromptTemplate) -> None:
        """Update the prompt template used for subsequent LLM calls.

        Args:
            prompt (BasePromptTemplate): New prompt template instance.

        Returns:
            None: The method mutates the stored prompt in place.

        Raises:
            TypeError: Propagated if the supplied prompt does not support the template interface.

        Examples:
            - Swap the prompt template at runtime
                ```python
                >>> from types import SimpleNamespace
                >>> from pydantic import BaseModel
                >>> from serapeum.core.prompts.base import PromptTemplate
                >>> from serapeum.core.structured_tools.text_completion_llm import TextCompletionLLM
                >>> from serapeum.llms.ollama import Ollama
                >>> LLM = Ollama(
                ...     model="llama3.1",
                ...     request_timeout=180,
                >>> )
                >>> class Item(BaseModel):
                ...     value: int
                >>> text_llm = TextCompletionLLM(
                ...     output_cls=Item,
                ...     prompt="Value?",
                ...     llm=LLM,
                ... )
                >>> text_llm.prompt = PromptTemplate("Next value?")
                >>> text_llm.prompt.get_template()
                'Next value?'

                ```

        See Also:
            TextCompletionLLM.validate_prompt: Performs validation when constructing instances.
        """
        self._prompt = prompt

    def __call__(
        self,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> BaseModel:
        """Execute the prompt synchronously and parse the structured response.

        Args:
            llm_kwargs (Optional[Dict[str, Any]]): Keyword arguments forwarded to the underlying
                LLM invocation (chat or completion paths).
            *args (Any): Positional arguments accepted for interface compatibility; unused.
            **kwargs (Any): Prompt variables applied when rendering the template.

        Returns:
            BaseModel: Parsed Pydantic object produced by the configured output parser.

        Raises:
            ValueError: If the parsed object does not match the declared `output_cls`.

        Examples:
            - Parse a JSON completion into a Pydantic model
                ```python
                >>> from types import SimpleNamespace
                >>> from pydantic import BaseModel
                >>> from serapeum.core.output_parsers.models import PydanticOutputParser
                >>> from serapeum.llms.ollama import Ollama
                >>> LLM = Ollama(
                ...     model="llama3.1",
                ...     request_timeout=180,
                >>> )
                >>> class Record(BaseModel):
                ...     value: int
                >>> text_llm = TextCompletionLLM(
                ...     output_parser=PydanticOutputParser(output_cls=Record),
                ...     prompt="Return an integer.",
                ...     llm=LLM,
                ... )
                >>> text_llm() # doctest: +SKIP
                Record(value=7)

                ```
            - Surface parser type mismatches
                ```python
                >>> from types import SimpleNamespace
                >>> from pydantic import BaseModel
                >>> from serapeum.core.output_parsers import BaseOutputParser
                >>> class Record(BaseModel):
                ...     value: int
                >>> class EchoParser(BaseOutputParser):
                ...     def parse(self, output: str):
                ...         return output
                >>> text_llm = TextCompletionLLM(
                ...     output_parser=EchoParser(),
                ...     output_cls=Record,
                ...     prompt="Return data.",
                ...     llm=LLM,
                ... )
                >>> text_llm()  # doctest: +SKIP
                Traceback (most recent call last):
                ...
                ValueError: Output parser returned <class 'str'> but expected <class '...Record'>

                ```

        See Also:
            TextCompletionLLM.acall: Asynchronous equivalent.
            TextCompletionLLM.validate_output_parser_cls: Ensures parser compatibility.
        """
        llm_kwargs = llm_kwargs or {}
        if self._llm.metadata.is_chat_model:
            messages = self._prompt.format_messages(llm=self._llm, **kwargs)
            messages = self._llm._extend_messages(messages)
            chat_response = self._llm.chat(messages, **llm_kwargs)

            raw_output = chat_response.message.content or ""
        else:
            formatted_prompt = self._prompt.format(llm=self._llm, **kwargs)

            response = self._llm.complete(formatted_prompt, **llm_kwargs)

            raw_output = response.text

        output = self._output_parser.parse(raw_output)
        if not isinstance(output, self._output_cls):
            raise ValueError(
                f"Output parser returned {type(output)} but expected {self._output_cls}"
            )
        return output

    async def acall(
        self,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> BaseModel:
        llm_kwargs = llm_kwargs or {}
        if self._llm.metadata.is_chat_model:
            messages = self._prompt.format_messages(llm=self._llm, **kwargs)
            messages = self._llm._extend_messages(messages)
            chat_response = await self._llm.achat(messages, **llm_kwargs)

            raw_output = chat_response.message.content or ""
        else:
            formatted_prompt = self._prompt.format(llm=self._llm, **kwargs)

            response = await self._llm.acomplete(formatted_prompt, **llm_kwargs)

            raw_output = response.text

        output = self._output_parser.parse(raw_output)
        if not isinstance(output, self._output_cls):
            raise ValueError(
                f"Output parser returned {type(output)} but expected {self._output_cls}"
            )
        return output
