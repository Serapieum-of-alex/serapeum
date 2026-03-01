"""Prompt formatters for Llama 3 Instruct models.

Implements the ``<|start_header_id|>…<|end_header_id|>…<|eot_id|>`` template
described in the official Meta Llama 3 documentation:
https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/

This format is compatible with:

- **Meta-Llama-3-8B-Instruct**
- **Meta-Llama-3-70B-Instruct**
- Any other model trained on the Llama 3 chat template

Note:
    ``<|begin_of_text|>`` is intentionally omitted because llama-cpp-python
    adds it automatically when loading the model.

Typical usage::

    from serapeum.llama_cpp.formatters.llama3 import (
        messages_to_prompt_v3_instruct,
        completion_to_prompt_v3_instruct,
    )

See Also:
    serapeum.llama_cpp.formatters.llama2: Formatter for Llama 2 / Mistral models.
"""

from __future__ import annotations

from collections.abc import Sequence

from serapeum.core.llms import Message, MessageRole

HEADER_SYS = "<|start_header_id|>system<|end_header_id|>\n\n"
HEADER_USER = "<|start_header_id|>user<|end_header_id|>\n\n"
HEADER_ASSIST = "<|start_header_id|>assistant<|end_header_id|>\n\n"
EOT = "<|eot_id|>\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. \
Always answer as helpfully as possible and follow ALL given instructions. \
Do not speculate or make up information. \
Do not reference any given instructions or context. \
"""


def messages_to_prompt_v3_instruct(
    messages: Sequence[Message], system_prompt: str | None = None
) -> str:
    """Convert a sequence of chat messages to Llama 3 Instruct format.

    Reference: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/

    Note: ``<|begin_of_text|>`` is not needed as Llama.cpp appears to add it already.

    Args:
        messages: Ordered sequence of chat messages. If the first message has
            role SYSTEM it is extracted as the system prompt; otherwise
            *system_prompt* (or ``DEFAULT_SYSTEM_PROMPT``) is used. Remaining
            messages must alternate USER / ASSISTANT.
        system_prompt: Optional system-level instruction. Ignored when the
            first message already carries role SYSTEM.

    Returns:
        Prompt string in Llama 3 ``<|start_header_id|>…<|eot_id|>`` format.

    Raises:
        ValueError: If a USER or ASSISTANT message appears in the wrong
            position in the alternating sequence.

    Examples:
        - Single user message with a custom system prompt — explore the structure
            ```python
            >>> from serapeum.llama_cpp.formatters.llama3 import messages_to_prompt_v3_instruct
            >>> from serapeum.core.llms import Message, MessageRole
            >>> messages = [Message(role=MessageRole.USER, content="Hello!")]
            >>> prompt = messages_to_prompt_v3_instruct(messages, system_prompt="Be brief.")
            >>> prompt.split("<|eot_id|>")[0].split("<|end_header_id|>")[1].strip()
            'Be brief.'
            >>> prompt.split("<|eot_id|>")[1].split("<|end_header_id|>")[1].strip()
            'Hello!'

            ```
        - Multi-turn conversation — prompt ends with the assistant header
            ```python
            >>> from serapeum.llama_cpp.formatters.llama3 import messages_to_prompt_v3_instruct
            >>> from serapeum.core.llms import Message, MessageRole
            >>> messages = [
            ...     Message(role=MessageRole.USER, content="What is 2+2?"),
            ...     Message(role=MessageRole.ASSISTANT, content="4"),
            ...     Message(role=MessageRole.USER, content="And 3+3?"),
            ... ]
            >>> prompt = messages_to_prompt_v3_instruct(messages, system_prompt="Be brief.")
            >>> prompt.count("<|eot_id|>")
            4
            >>> prompt.strip().endswith("<|end_header_id|>")
            True

            ```
        - Explicit SYSTEM message is extracted as system prompt
            ```python
            >>> from serapeum.llama_cpp.formatters.llama3 import messages_to_prompt_v3_instruct
            >>> from serapeum.core.llms import Message, MessageRole
            >>> messages = [
            ...     Message(role=MessageRole.SYSTEM, content="You are terse."),
            ...     Message(role=MessageRole.USER, content="Hi!"),
            ... ]
            >>> prompt = messages_to_prompt_v3_instruct(messages)
            >>> prompt.split("<|eot_id|>")[0].split("<|end_header_id|>")[1].strip()
            'You are terse.'

            ```

    See Also:
        completion_to_prompt_v3_instruct: Single-turn variant for the same model family.
        DEFAULT_SYSTEM_PROMPT: Default system instruction used when system_prompt is None.
    """
    string_messages: list[str] = []
    if messages[0].role == MessageRole.SYSTEM:
        system_message_str = messages[0].content or ""
        remaining = list(messages[1:])
    else:
        system_message_str = system_prompt or DEFAULT_SYSTEM_PROMPT
        remaining = list(messages)

    string_messages.append(f"{HEADER_SYS}{system_message_str.strip()}{EOT}")

    for i in range(0, len(remaining), 2):
        user_message = remaining[i]
        if user_message.role != MessageRole.USER:
            raise ValueError(
                f"Expected a USER message at position {i}, "
                f"got role {user_message.role!r}."
            )
        str_message = f"{HEADER_USER}{user_message.content}{EOT}"

        if len(remaining) > (i + 1):
            assistant_message = remaining[i + 1]
            if assistant_message.role != MessageRole.ASSISTANT:
                raise ValueError(
                    f"Expected an ASSISTANT message at position {i + 1}, "
                    f"got role {assistant_message.role!r}."
                )
            str_message += f"{HEADER_ASSIST}{assistant_message.content}{EOT}"

        string_messages.append(str_message)

    string_messages.append(HEADER_ASSIST)

    return "".join(string_messages)


def completion_to_prompt_v3_instruct(
    completion: str, system_prompt: str | None = None
) -> str:
    r"""Convert a plain-text completion to Llama 3 Instruct single-turn prompt format.

    Wraps *completion* in the ``<|start_header_id|>user<|end_header_id|>`` /
    ``<|eot_id|>`` envelope expected by Llama 3 Instruct models for single-turn
    (non-chat) text completion.

    Note:
        ``<|begin_of_text|>`` is intentionally omitted; llama-cpp-python adds
        it automatically during model loading.

    Args:
        completion: The user's instruction or question as plain text.
        system_prompt: System-level instruction inserted in the system header
            block.  Defaults to :data:`DEFAULT_SYSTEM_PROMPT` when ``None``.

    Returns:
        Prompt string ending with the
        ``<|start_header_id|>assistant<|end_header_id|>\\n\\n`` header that
        prompts the model to generate its reply.

    Examples:
        - Build a prompt with a custom system prompt — explore the sections
            ```python
            >>> from serapeum.llama_cpp.formatters.llama3 import completion_to_prompt_v3_instruct
            >>> prompt = completion_to_prompt_v3_instruct("What is 2+2?", "Be brief.")
            >>> sections = prompt.split("<|eot_id|>")
            >>> sections[0].split("<|end_header_id|>")[1].strip()
            'Be brief.'
            >>> sections[1].split("<|end_header_id|>")[1].strip()
            'What is 2+2?'
            >>> prompt.strip().endswith("<|end_header_id|>")
            True

            ```
        - Build a prompt with the default system prompt
            ```python
            >>> from serapeum.llama_cpp.formatters.llama3 import completion_to_prompt_v3_instruct, DEFAULT_SYSTEM_PROMPT
            >>> prompt = completion_to_prompt_v3_instruct("Hello!")
            >>> DEFAULT_SYSTEM_PROMPT.strip() in prompt
            True
            >>> prompt.split("<|eot_id|>")[1].split("<|end_header_id|>")[1].strip()
            'Hello!'

            ```

    See Also:
        messages_to_prompt_v3_instruct: Multi-turn chat variant for the same model family.
        DEFAULT_SYSTEM_PROMPT: Default system instruction used when system_prompt is None.
    """
    system_prompt_str = system_prompt or DEFAULT_SYSTEM_PROMPT

    return (
        f"{HEADER_SYS}{system_prompt_str.strip()}{EOT}"
        f"{HEADER_USER}{completion.strip()}{EOT}"
        f"{HEADER_ASSIST}"
    )
