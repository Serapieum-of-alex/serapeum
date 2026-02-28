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
    """
    Convert completion instruction string to Llama 3 Instruct format.

    Reference: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/

    Note: `<|begin_of_text|>` is not needed as Llama.cpp appears to add it already.
    """
    system_prompt_str = system_prompt or DEFAULT_SYSTEM_PROMPT

    return (
        f"{HEADER_SYS}{system_prompt_str.strip()}{EOT}"
        f"{HEADER_USER}{completion.strip()}{EOT}"
        f"{HEADER_ASSIST}"
    )
