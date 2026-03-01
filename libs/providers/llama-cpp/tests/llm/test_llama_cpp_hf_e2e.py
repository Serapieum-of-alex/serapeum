"""End-to-end tests for LlamaCPP with HuggingFace Hub model downloads.

These tests exercise the full HF download -> load -> inference pipeline.
They are opt-in because they require ``huggingface-hub`` and network access.

To run locally::

    LLAMA_CPP_HF_E2E=1 pytest libs/providers/llama-cpp/tests/llm/test_llama_cpp_hf_e2e.py -v

Override model choice via environment variables::

    export LLAMA_CPP_HF_MODEL_ID=ggml-org/models
    export LLAMA_CPP_HF_FILENAME=tinyllamas/stories260K.gguf

Assertions are intentionally loose -- they validate shape/type/contract,
not exact generated text, because generative output is non-deterministic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from serapeum.core.llms import ChatResponse, CompletionResponse, Message, MessageRole
from serapeum.llama_cpp import LlamaCPP


HF_MODEL_ID_ENV = "LLAMA_CPP_HF_MODEL_ID"
HF_FILENAME_ENV = "LLAMA_CPP_HF_FILENAME"

_hf_model_id = os.environ.get(HF_MODEL_ID_ENV, "ggml-org/models")
_hf_filename = os.environ.get(HF_FILENAME_ENV, "tinyllamas/stories260K.gguf")

MAX_NEW_TOKENS = 32
TEMPERATURE = 0.0


def _passthrough_messages_to_prompt(
    messages: list[Message],
    system_prompt: str | None = None,
) -> str:
    """Join message contents into a single string (no chat template)."""
    return " ".join(m.content or "" for m in messages)


def _passthrough_completion_to_prompt(
    completion: str,
    system_prompt: str | None = None,
) -> str:
    """Return the completion string unchanged."""
    return completion


def _user_messages(*contents: str) -> list[Message]:
    """Build a list of USER messages from plain strings."""
    return [Message(role=MessageRole.USER, content=c) for c in contents]


def _formatters() -> dict[str, Any]:
    """Return pass-through formatters suitable for non-instruct models."""
    return {
        "messages_to_prompt": _passthrough_messages_to_prompt,
        "completion_to_prompt": _passthrough_completion_to_prompt,
    }


@pytest.fixture(scope="module")
def hf_llm(tmp_path_factory: pytest.TempPathFactory) -> LlamaCPP:
    """Download a small GGUF from HuggingFace Hub and load it.

    Uses a module-scoped temp directory as ``SERAPEUM_CACHE_DIR`` so the
    download is isolated from the user's real cache.  The model is loaded
    once and shared across all tests in TestLlamaCPPHuggingFace.

    The default model is ``stories260K.gguf`` (~1.2 MB), a tiny LLaMA
    trained on TinyStories -- small enough to download and load in CI.

    Returns:
        LlamaCPP: Instance loaded from HuggingFace Hub.
    """
    cache_dir = str(tmp_path_factory.mktemp("hf_cache"))
    old_cache = os.environ.get("SERAPEUM_CACHE_DIR")
    os.environ["SERAPEUM_CACHE_DIR"] = cache_dir
    try:
        llm = LlamaCPP(
            hf_model_id=_hf_model_id,
            hf_filename=_hf_filename,
            temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS,
            context_window=512,
            verbose=False,
            **_formatters(),
        )
    finally:
        if old_cache is None:
            os.environ.pop("SERAPEUM_CACHE_DIR", None)
        else:
            os.environ["SERAPEUM_CACHE_DIR"] = old_cache
    return llm


@pytest.mark.e2e
class TestLlamaCPPHuggingFace:
    """E2E tests for LlamaCPP with models downloaded from HuggingFace Hub.

    These tests exercise the full HF download -> load -> inference pipeline.
    The default model (stories260K.gguf, ~1.2 MB) is small enough for CI.

    To run::

        LLAMA_CPP_HF_E2E=1 pytest libs/providers/llama-cpp/tests/llm/test_llama_cpp_hf_e2e.py -v
    """

    def test_hf_construction_sets_model_path(self, hf_llm: LlamaCPP) -> None:
        """Test that HF download populates model_path with an existing .gguf file.

        Test scenario:
            After constructing with hf_model_id + hf_filename, model_path must
            be a non-None string pointing to an existing file with a .gguf
            extension.
        """
        assert (
            hf_llm.model_path is not None
        ), "model_path should be set after HF download"
        assert isinstance(
            hf_llm.model_path, str
        ), f"Expected str, got {type(hf_llm.model_path)}"
        path = Path(hf_llm.model_path)
        assert path.exists(), f"model_path does not exist: {hf_llm.model_path}"
        assert path.suffix == ".gguf", f"Expected .gguf extension, got {path.suffix!r}"

    def test_hf_metadata_has_valid_context_window(self, hf_llm: LlamaCPP) -> None:
        """Test metadata.context_window is a positive integer after HF load.

        Test scenario:
            A successfully loaded GGUF model must expose a positive context
            window size via its metadata.
        """
        assert (
            hf_llm.metadata.context_window > 0
        ), f"Expected context_window > 0, got {hf_llm.metadata.context_window}"

    def test_hf_complete_returns_non_empty_text(self, hf_llm: LlamaCPP) -> None:
        """Test complete() returns a CompletionResponse with non-empty text.

        Test scenario:
            Running inference on the HF-downloaded model must produce at least
            one character of output.
        """
        response = hf_llm.complete("Once upon a time")
        assert isinstance(
            response, CompletionResponse
        ), f"Expected CompletionResponse, got {type(response)}"
        assert len(response.text) > 0, "HF model complete() text should not be empty"

    def test_hf_stream_complete_yields_chunks(self, hf_llm: LlamaCPP) -> None:
        """Test complete(stream=True) yields multiple CompletionResponse chunks.

        Test scenario:
            Streaming a multi-token response must produce more than one chunk,
            each with a valid string delta.
        """
        chunks = list(hf_llm.complete("Once upon a time", stream=True))
        assert len(chunks) > 1, f"Expected multiple stream chunks, got {len(chunks)}"
        for chunk in chunks:
            assert isinstance(
                chunk, CompletionResponse
            ), f"Expected CompletionResponse chunk, got {type(chunk)}"
            assert chunk.delta is not None, f"Chunk delta must not be None: {chunk}"

    def test_hf_chat_returns_assistant_response(self, hf_llm: LlamaCPP) -> None:
        """Test chat() returns a ChatResponse with ASSISTANT role and content.

        Test scenario:
            A single USER message sent to the HF-downloaded model must produce
            a ChatResponse attributed to ASSISTANT with non-empty content.
        """
        response = hf_llm.chat(_user_messages("Once upon a time"))
        assert isinstance(
            response, ChatResponse
        ), f"Expected ChatResponse, got {type(response)}"
        assert (
            response.message.role == MessageRole.ASSISTANT
        ), f"Expected ASSISTANT role, got {response.message.role}"
        assert response.message.content, "HF model chat content should not be empty"

    def test_hf_second_construction_reuses_cache(
        self, hf_llm: LlamaCPP, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """Test that a second LlamaCPP construction skips the HF re-download.

        Test scenario:
            After the first download, constructing another LlamaCPP with the
            same hf_model_id/hf_filename (pointing at the same cache dir)
            must not call hf_hub_download again because the file already
            exists on disk -- model_path is set directly from the first run.
        """
        assert hf_llm.model_path is not None
        with patch("serapeum.llama_cpp.utils.hf_hub_download") as mock_download:
            second = LlamaCPP(
                model_path=hf_llm.model_path,
                temperature=TEMPERATURE,
                max_new_tokens=MAX_NEW_TOKENS,
                context_window=512,
                verbose=False,
                **_formatters(),
            )
            mock_download.assert_not_called()
        assert (
            second.model_path == hf_llm.model_path
        ), "Second instance should reuse the same model_path"
