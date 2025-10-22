from typing import Any, Dict, Optional, Type, Union, Tuple

from pydantic import BaseModel
from serapeum.core.llm.base import LLM
from serapeum.core.output_parsers.models import PydanticOutputParser
from serapeum.core.prompts.base import BasePromptTemplate, PromptTemplate
from serapeum.core.configs.configs import Configs
from serapeum.core.structured_tools.models import BasePydanticProgram
from serapeum.core.output_parsers import BaseOutputParser


class TextCompletionLLM(BasePydanticProgram[BaseModel]):
    """
    LLM Text Completion Program.

    Uses generic LLM text completion + an output parser to generate a structured output.

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
        self._output_parser, self._output_cls = self.validate_output_parser_cls(output_parser, output_cls)
        self._llm = self.validate_llm(llm)
        self._prompt = self.validate_prompt(prompt)
        self._verbose = verbose
        self._prompt.output_parser = output_parser

    @staticmethod
    def validate_prompt(prompt: Union[BasePromptTemplate, str]) -> BasePromptTemplate:
        if not isinstance(prompt, (BasePromptTemplate, str)):
            raise ValueError(
                "prompt must be an instance of BasePromptTemplate or str."
            )
        if isinstance(prompt, str):
            prompt = PromptTemplate(prompt)
        return prompt

    @staticmethod
    def validate_llm(llm: LLM) -> LLM:
        llm = llm or Configs.llm  # type: ignore
        if llm is None:
            raise AssertionError("llm must be provided or set in Configs.")
        return llm

    @staticmethod
    def validate_output_parser_cls(
        output_parser: BaseOutputParser, output_cls: Type[BaseModel]
    ) -> Tuple[BaseOutputParser, Type[BaseModel]]:
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
        return self._output_cls

    @property
    def prompt(self) -> BasePromptTemplate:
        return self._prompt

    @prompt.setter
    def prompt(self, prompt: BasePromptTemplate) -> None:
        self._prompt = prompt

    def __call__(
        self,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> BaseModel:
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
