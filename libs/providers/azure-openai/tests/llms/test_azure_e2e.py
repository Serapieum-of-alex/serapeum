"""End-to-end tests for Azure OpenAI Completions and Responses classes.

These tests hit a real Azure OpenAI endpoint and require the following
environment variables to be set:

- ``AZURE_OPENAI_API_KEY`` or ``use_azure_ad=True`` with credentials
- ``AZURE_OPENAI_ENDPOINT`` (e.g. ``https://YOUR_RESOURCE.openai.azure.com/``)
- ``OPENAI_API_VERSION`` (e.g. ``2024-02-01``)
- ``AZURE_OPENAI_COMPLETIONS_ENGINE`` — deployment name for a Chat Completions model
- ``AZURE_OPENAI_RESPONSES_ENGINE`` — deployment name for a Responses-capable model

Skip markers ensure these tests are excluded from regular CI runs.
"""

from __future__ import annotations

import os

import pytest

from serapeum.azure_openai import Completions, Responses
from serapeum.core.llms import Message, TextChunk


def _get_env(name: str) -> str:
    """Get a required environment variable or skip the test."""
    value = os.getenv(name)
    if not value:
        pytest.skip(f"Environment variable {name} is not set")
    return value


@pytest.fixture(scope="module")
def completions_llm() -> Completions:
    """Create a Completions instance from environment variables.

    Returns:
        Completions: An Azure OpenAI Completions LLM connected to a real deployment.
    """
    return Completions(
        engine=_get_env("AZURE_OPENAI_COMPLETIONS_ENGINE"),
        azure_endpoint=_get_env("AZURE_OPENAI_ENDPOINT"),
        api_key=_get_env("AZURE_OPENAI_API_KEY"),
        api_version=_get_env("OPENAI_API_VERSION"),
    )


@pytest.fixture(scope="module")
def responses_llm() -> Responses:
    """Create a Responses instance from environment variables.

    Returns:
        Responses: An Azure OpenAI Responses LLM connected to a real deployment.
    """
    return Responses(
        engine=_get_env("AZURE_OPENAI_RESPONSES_ENGINE"),
        azure_endpoint=_get_env("AZURE_OPENAI_ENDPOINT"),
        api_key=_get_env("AZURE_OPENAI_API_KEY"),
        api_version=_get_env("OPENAI_API_VERSION"),
    )


@pytest.mark.e2e
class TestCompletionsE2E:
    """End-to-end tests for Azure Completions (Chat Completions API)."""

    def test_complete_non_streaming(self, completions_llm: Completions) -> None:
        """Non-streaming complete returns a non-empty response.

        Test scenario:
            A simple prompt should produce a CompletionResponse with
            non-empty text from the real Azure deployment.
        """
        response = completions_llm.complete("Say hello in one word.")
        assert response.text, "Expected non-empty response text"
        assert len(response.text) > 0

    def test_complete_streaming(self, completions_llm: Completions) -> None:
        """Streaming complete yields at least one chunk.

        Test scenario:
            A streaming call should yield incremental CompletionResponse
            chunks that accumulate into a non-empty final response.
        """
        response_gen = completions_llm.complete("Say hello in one word.", stream=True)
        responses = list(response_gen)
        assert len(responses) > 0, "Expected at least one streaming chunk"
        assert responses[-1].text, "Final chunk should have accumulated text"

    def test_chat_non_streaming(self, completions_llm: Completions) -> None:
        """Non-streaming chat returns a ChatResponse with assistant message.

        Test scenario:
            A single user message should produce a ChatResponse with
            role='assistant' and non-empty content.
        """
        message = Message(
            role="user", chunks=[TextChunk(content="Say hello in one word.")]
        )
        response = completions_llm.chat([message])
        assert (
            response.message.role == "assistant"
        ), f"Expected role 'assistant', got '{response.message.role}'"
        assert response.message.content, "Expected non-empty message content"

    def test_chat_streaming(self, completions_llm: Completions) -> None:
        """Streaming chat yields at least one ChatResponse chunk.

        Test scenario:
            A streaming chat call should yield ChatResponse chunks with
            the final chunk having non-empty assistant content.
        """
        message = Message(
            role="user", chunks=[TextChunk(content="Say hello in one word.")]
        )
        chat_gen = completions_llm.chat([message], stream=True)
        chat_responses = list(chat_gen)
        assert len(chat_responses) > 0, "Expected at least one streaming chunk"
        assert chat_responses[
            -1
        ].message.content, "Final chunk should have accumulated content"

    @pytest.mark.asyncio
    async def test_acomplete_non_streaming(self, completions_llm: Completions) -> None:
        """Async non-streaming complete returns a CompletionResponse.

        Test scenario:
            acomplete() should return a valid response from the real endpoint.
        """
        response = await completions_llm.acomplete("Say hello in one word.")
        assert response.text, "Expected non-empty response text"

    @pytest.mark.asyncio
    async def test_achat_non_streaming(self, completions_llm: Completions) -> None:
        """Async non-streaming chat returns a ChatResponse.

        Test scenario:
            achat() should return an assistant message from the real endpoint.
        """
        message = Message(
            role="user", chunks=[TextChunk(content="Say hello in one word.")]
        )
        response = await completions_llm.achat([message])
        assert response.message.role == "assistant"
        assert response.message.content, "Expected non-empty message content"

    @pytest.mark.asyncio
    async def test_acomplete_streaming(self, completions_llm: Completions) -> None:
        """Async streaming complete yields CompletionResponse chunks.

        Test scenario:
            acomplete(stream=True) should yield at least one chunk.
        """
        response_gen = await completions_llm.acomplete(
            "Say hello in one word.", stream=True
        )
        responses = [item async for item in response_gen]
        assert len(responses) > 0, "Expected at least one streaming chunk"
        assert responses[-1].text, "Final chunk should have accumulated text"

    @pytest.mark.asyncio
    async def test_achat_streaming(self, completions_llm: Completions) -> None:
        """Async streaming chat yields ChatResponse chunks.

        Test scenario:
            achat(stream=True) should yield at least one chunk.
        """
        message = Message(
            role="user", chunks=[TextChunk(content="Say hello in one word.")]
        )
        chat_gen = await completions_llm.achat([message], stream=True)
        chat_responses = [item async for item in chat_gen]
        assert len(chat_responses) > 0, "Expected at least one streaming chunk"
        assert chat_responses[
            -1
        ].message.content, "Final chunk should have accumulated content"


@pytest.mark.e2e
class TestResponsesE2E:
    """End-to-end tests for Azure Responses (Responses API)."""

    def test_complete_non_streaming(self, responses_llm: Responses) -> None:
        """Non-streaming complete returns a non-empty response.

        Test scenario:
            A simple prompt should produce a CompletionResponse with
            non-empty text from the real Azure Responses deployment.
        """
        response = responses_llm.complete("Say hello in one word.")
        assert response.text, "Expected non-empty response text"

    def test_complete_streaming(self, responses_llm: Responses) -> None:
        """Streaming complete yields at least one chunk.

        Test scenario:
            A streaming call should yield incremental response chunks.
        """
        response_gen = responses_llm.complete("Say hello in one word.", stream=True)
        responses = list(response_gen)
        assert len(responses) > 0, "Expected at least one streaming chunk"
        assert responses[-1].text, "Final chunk should have accumulated text"

    def test_chat_non_streaming(self, responses_llm: Responses) -> None:
        """Non-streaming chat returns a ChatResponse with assistant message.

        Test scenario:
            A single user message should produce a ChatResponse with
            role='assistant' and non-empty content.
        """
        message = Message(
            role="user", chunks=[TextChunk(content="Say hello in one word.")]
        )
        response = responses_llm.chat([message])
        assert (
            response.message.role == "assistant"
        ), f"Expected role 'assistant', got '{response.message.role}'"
        assert response.message.content, "Expected non-empty message content"

    def test_chat_streaming(self, responses_llm: Responses) -> None:
        """Streaming chat yields at least one ChatResponse chunk.

        Test scenario:
            A streaming chat call should yield ChatResponse chunks with
            non-empty final content.
        """
        message = Message(
            role="user", chunks=[TextChunk(content="Say hello in one word.")]
        )
        chat_gen = responses_llm.chat([message], stream=True)
        chat_responses = list(chat_gen)
        assert len(chat_responses) > 0, "Expected at least one streaming chunk"
        assert chat_responses[
            -1
        ].message.content, "Final chunk should have accumulated content"

    @pytest.mark.asyncio
    async def test_acomplete_non_streaming(self, responses_llm: Responses) -> None:
        """Async non-streaming complete returns a CompletionResponse.

        Test scenario:
            acomplete() should return a valid response from the real endpoint.
        """
        response = await responses_llm.acomplete("Say hello in one word.")
        assert response.text, "Expected non-empty response text"

    @pytest.mark.asyncio
    async def test_achat_non_streaming(self, responses_llm: Responses) -> None:
        """Async non-streaming chat returns a ChatResponse.

        Test scenario:
            achat() should return an assistant message from the real endpoint.
        """
        message = Message(
            role="user", chunks=[TextChunk(content="Say hello in one word.")]
        )
        response = await responses_llm.achat([message])
        assert response.message.role == "assistant"
        assert response.message.content, "Expected non-empty message content"

    @pytest.mark.asyncio
    async def test_acomplete_streaming(self, responses_llm: Responses) -> None:
        """Async streaming complete yields CompletionResponse chunks.

        Test scenario:
            acomplete(stream=True) should yield at least one chunk.
        """
        response_gen = await responses_llm.acomplete(
            "Say hello in one word.", stream=True
        )
        responses = [item async for item in response_gen]
        assert len(responses) > 0, "Expected at least one streaming chunk"
        assert responses[-1].text, "Final chunk should have accumulated text"

    @pytest.mark.asyncio
    async def test_achat_streaming(self, responses_llm: Responses) -> None:
        """Async streaming chat yields ChatResponse chunks.

        Test scenario:
            achat(stream=True) should yield at least one chunk.
        """
        message = Message(
            role="user", chunks=[TextChunk(content="Say hello in one word.")]
        )
        chat_gen = await responses_llm.achat([message], stream=True)
        chat_responses = [item async for item in chat_gen]
        assert len(chat_responses) > 0, "Expected at least one streaming chunk"
        assert chat_responses[
            -1
        ].message.content, "Final chunk should have accumulated content"
