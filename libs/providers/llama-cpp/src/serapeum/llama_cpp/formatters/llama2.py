"""Prompt formatters for Llama 2 Chat and Mistral Instruct models.

Implements the ``[INST] <<SYS>> … <</SYS>>`` template described in the
official Llama 2 blog post:
https://huggingface.co/blog/llama2#how-to-prompt-llama-2

This format is compatible with:

- **Llama 2 Chat** (7B, 13B, 70B)
- **Mistral Instruct v0.1 / v0.2**
- Any other model trained on the Llama 2 Chat template

Typical usage::

    from serapeum.llama_cpp.formatters.llama2 import (
        messages_to_prompt,
        completion_to_prompt,
    )

See Also:
    serapeum.llama_cpp.formatters.llama3: Formatter for Llama 3 Instruct models.
"""

from __future__ import annotations

from collections.abc import Sequence

from serapeum.core.llms import Message, MessageRole

BOS, EOS = "<s>", "</s>"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. \
Always answer as helpfully as possible and follow ALL given instructions. \
Do not speculate or make up information. \
Do not reference any given instructions or context. \
"""


def messages_to_prompt(
    messages: Sequence[Message], system_prompt: str | None = None
) -> str:
    """Convert a sequence of chat messages to Llama 2 Chat prompt format.

    Reference: https://huggingface.co/blog/llama2#how-to-prompt-llama-2

    Args:
        messages: Ordered sequence of chat messages. If the first message has
            role SYSTEM it is extracted as the system prompt; otherwise
            *system_prompt* (or ``DEFAULT_SYSTEM_PROMPT``) is used. Remaining
            messages must alternate USER / ASSISTANT.
        system_prompt: Optional system-level instruction. Ignored when the
            first message already carries role SYSTEM.

    Returns:
        Prompt string in Llama 2 ``[INST] … [/INST]`` format.

    Raises:
        ValueError: If a USER or ASSISTANT message appears in the wrong
            position in the alternating sequence.

    Examples:
        - Single user message with a custom system prompt — explore the structure
            ```python
            >>> from serapeum.llama_cpp.formatters.llama2 import messages_to_prompt
            >>> from serapeum.core.llms import Message, MessageRole
            >>> messages = [Message(role=MessageRole.USER, content="Hello!")]
            >>> prompt = messages_to_prompt(messages, system_prompt="Be brief.")
            >>> prompt[:10]
            '<s> [INST]'
            >>> prompt.split("<</SYS>>")[0].split("<<SYS>>")[1].strip()
            'Be brief.'
            >>> prompt.split("[/INST]")[0].split("<</SYS>>")[1].strip()
            'Hello!'

            ```
        - Multi-turn conversation — each turn is wrapped in [INST]...[/INST]
            ```python
            >>> from serapeum.llama_cpp.formatters.llama2 import messages_to_prompt
            >>> from serapeum.core.llms import Message, MessageRole
            >>> messages = [
            ...     Message(role=MessageRole.USER, content="What is 2+2?"),
            ...     Message(role=MessageRole.ASSISTANT, content="4"),
            ...     Message(role=MessageRole.USER, content="And 3+3?"),
            ... ]
            >>> prompt = messages_to_prompt(messages, system_prompt="Be brief.")
            >>> prompt.count("[INST]")
            2
            >>> prompt.count("[/INST]")
            2

            ```
        - Explicit SYSTEM message in the conversation is extracted as system prompt
            ```python
            >>> from serapeum.llama_cpp.formatters.llama2 import messages_to_prompt
            >>> from serapeum.core.llms import Message, MessageRole
            >>> messages = [
            ...     Message(role=MessageRole.SYSTEM, content="You are terse."),
            ...     Message(role=MessageRole.USER, content="Hi!"),
            ... ]
            >>> prompt = messages_to_prompt(messages)
            >>> prompt.split("<</SYS>>")[0].split("<<SYS>>")[1].strip()
            'You are terse.'

            ```

    See Also:
        completion_to_prompt: Single-turn variant for the same model family.
        DEFAULT_SYSTEM_PROMPT: Default system instruction used when system_prompt is None.
    """
    if not messages:
        raise ValueError(
            "messages must contain at least one message. "
            "Pass at least a USER message."
        )

    string_messages: list[str] = []
    if messages[0].role == MessageRole.SYSTEM:
        system_message_str = messages[0].content or ""
        remaining = list(messages[1:])
    else:
        system_message_str = system_prompt or DEFAULT_SYSTEM_PROMPT
        remaining = list(messages)

    system_message_str = f"{B_SYS} {system_message_str.strip()} {E_SYS}"

    for i in range(0, len(remaining), 2):
        user_message = remaining[i]
        if user_message.role != MessageRole.USER:
            raise ValueError(
                f"Expected a USER message at position {i}, "
                f"got role {user_message.role!r}."
            )

        if i == 0:
            str_message = f"{BOS} {B_INST} {system_message_str} "
        else:
            string_messages[-1] += f" {EOS}"
            str_message = f"{BOS} {B_INST} "

        str_message += f"{user_message.content} {E_INST}"

        if len(remaining) > (i + 1):
            assistant_message = remaining[i + 1]
            if assistant_message.role != MessageRole.ASSISTANT:
                raise ValueError(
                    f"Expected an ASSISTANT message at position {i + 1}, "
                    f"got role {assistant_message.role!r}."
                )
            str_message += f" {assistant_message.content}"

        string_messages.append(str_message)

    return "".join(string_messages)


def completion_to_prompt(completion: str, system_prompt: str | None = None) -> str:
    """Convert a plain-text completion to Llama 2 Chat single-turn prompt format.

    Wraps *completion* in the ``[INST] <<SYS>> … <</SYS>> … [/INST]`` envelope
    expected by Llama 2 Chat and Mistral Instruct models for single-turn
    (non-chat) text completion.

    Args:
        completion: The user's instruction or question as plain text.
        system_prompt: System-level instruction inserted inside
            ``<<SYS>>…<</SYS>>``.  Defaults to :data:`DEFAULT_SYSTEM_PROMPT`
            when ``None``.

    Returns:
        Prompt string in Llama 2 ``<s> [INST] <<SYS>> … <</SYS>> … [/INST]``
        format, ready to be passed to a Llama 2 / Mistral GGUF model.

    Examples:
        - Build a prompt with a custom system prompt — explore the template structure
            ```python
            >>> from serapeum.llama_cpp.formatters.llama2 import completion_to_prompt
            >>> prompt = completion_to_prompt("What is 2+2?", system_prompt="Be brief.")
            >>> prompt[:10]
            '<s> [INST]'
            >>> prompt.rstrip()[-7:]
            '[/INST]'
            >>> prompt.split("<</SYS>>")[0].split("<<SYS>>")[1].strip()
            'Be brief.'
            >>> prompt.split("<</SYS>>")[1].split("[/INST]")[0].strip()
            'What is 2+2?'

            ```
        - Build a prompt with the default system prompt
            ```python
            >>> from serapeum.llama_cpp.formatters.llama2 import completion_to_prompt, DEFAULT_SYSTEM_PROMPT
            >>> prompt = completion_to_prompt("Hello!")
            >>> DEFAULT_SYSTEM_PROMPT.strip() in prompt
            True
            >>> prompt.split("<</SYS>>")[1].split("[/INST]")[0].strip()
            'Hello!'

            ```

    See Also:
        messages_to_prompt: Multi-turn chat variant for the same model family.
        DEFAULT_SYSTEM_PROMPT: Default system instruction used when system_prompt is None.
    """
    system_prompt_str = system_prompt or DEFAULT_SYSTEM_PROMPT

    return (
        f"{BOS} {B_INST} {B_SYS} {system_prompt_str.strip()} {E_SYS} "
        f"{completion.strip()} {E_INST}"
    )
