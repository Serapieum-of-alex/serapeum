from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
import logging
import math
import requests
from tqdm import tqdm
from serapeum.core.llms import Message, MessageRole

__all__ = [
    "messages_to_prompt",
    "completion_to_prompt",
    "messages_to_prompt_v3_instruct",
    "completion_to_prompt_v3_instruct",
]

logger = logging.getLogger(__name__)


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


HEADER_SYS = "<|start_header_id|>system<|end_header_id|>\n\n"
HEADER_USER = "<|start_header_id|>user<|end_header_id|>\n\n"
HEADER_ASSIST = "<|start_header_id|>assistant<|end_header_id|>\n\n"
EOT = "<|eot_id|>\n"


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


def _fetch_model_file(model_url: str, model_path: Path) -> None:
    """Download a model file from a URL to a local path with progress reporting.

    Streams the response in 1 MiB chunks to avoid loading the entire file into
    memory. Validates that the server reports a ``Content-Length`` of at least
    1 MB before writing any bytes. On any failure the partially written file is
    removed and the original exception is re-raised unchanged.

    Args:
        model_url: Fully-qualified URL of the GGUF model file to download.
        model_path: Destination path where the model file will be written.
            The parent directory must already exist.

    Raises:
        ValueError: If the server's ``Content-Length`` header is absent or
            reports fewer than 1 000 000 bytes, indicating the response is
            not a valid model file.
        requests.exceptions.HTTPError: If the server returns a 4xx or 5xx
            HTTP status code.
        requests.exceptions.ConnectionError: If a network-level error occurs
            before or during the download.
        OSError: If ``model_path`` cannot be opened for writing.

    Examples:
        - Download a GGUF model file to a local directory
            ```python
            >>> from pathlib import Path
            >>> _fetch_model_file(  # doctest: +SKIP
            ...     "https://example.com/model.gguf",
            ...     Path("/models/model.gguf"),
            ... )

            ```
        - Partial files are cleaned up automatically on failure
            ```python
            >>> from pathlib import Path
            >>> from unittest.mock import patch, MagicMock
            >>> mock_resp = MagicMock()
            >>> mock_resp.raise_for_status.return_value = None
            >>> mock_resp.headers.get.return_value = "500"  # < 1 MB → ValueError
            >>> mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            >>> mock_resp.__exit__ = MagicMock(return_value=False)
            >>> dest = Path("/tmp/model.gguf")
            >>> with patch("serapeum.llama_cpp.utils.requests.get", return_value=mock_resp):
            ...     try:
            ...         _fetch_model_file("https://example.com/model.gguf", dest)
            ...     except ValueError as exc:
            ...         print(exc)
            Content-Length is 500 bytes; expected at least 1 MB

            ```

    See Also:
        tqdm: Progress bar library used to display download progress.
        requests.Response.iter_content: Underlying streaming iterator.
    """
    logger.info("Downloading %s to %s", model_url, model_path)
    try:
        with requests.get(model_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("Content-Length") or 0)
            if total_size < 1_000_000:
                raise ValueError(
                    f"Content-Length is {total_size} bytes; expected at least 1 MB"
                )
            logger.info("Total size: %.2f MB", total_size / 1_000_000)
            chunk_size = 1024 * 1024  # 1 MB
            with model_path.open("wb") as file:
                for chunk in tqdm(
                    r.iter_content(chunk_size=chunk_size),
                    total=math.ceil(total_size / chunk_size),
                    unit="MB",
                ):
                    file.write(chunk)
    except Exception:
        logger.exception("Download failed, removing partial file at %s", model_path)
        model_path.unlink(missing_ok=True)
        raise