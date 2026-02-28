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
    """
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
    system_prompt_str = system_prompt or DEFAULT_SYSTEM_PROMPT

    return (
        f"{BOS} {B_INST} {B_SYS} {system_prompt_str.strip()} {E_SYS} "
        f"{completion.strip()} {E_INST}"
    )
