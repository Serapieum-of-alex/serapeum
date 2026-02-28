"""Unit tests for serapeum.llama_cpp.formatters.llama2."""
from __future__ import annotations

import pytest
from serapeum.core.llms import Message, MessageRole
from serapeum.llama_cpp.formatters.llama2 import (
    BOS,
    EOS,
    B_INST,
    E_INST,
    B_SYS,
    E_SYS,
    DEFAULT_SYSTEM_PROMPT,
    completion_to_prompt,
    messages_to_prompt,
)


def _user(content: str) -> Message:
    return Message(role=MessageRole.USER, content=content)


def _assist(content: str) -> Message:
    return Message(role=MessageRole.ASSISTANT, content=content)


def _system(content: str) -> Message:
    return Message(role=MessageRole.SYSTEM, content=content)


def _sys_block(sys: str) -> str:
    """Return the formatted system block that appears in the prompt."""
    return f"{B_SYS} {sys.strip()} {E_SYS}"


@pytest.mark.unit
class TestMessagesToPrompt:
    """Tests for messages_to_prompt (Llama 2 Chat format)."""

    def test_single_user_with_explicit_system_prompt(self) -> None:
        """Test basic single-turn prompt with an explicit system string.

        Test scenario:
            A single USER message and an explicit system_prompt should produce
            one ``<s> [INST] … [/INST]`` block containing the system block
            and the user content.
        """
        messages = [_user("Hello")]
        result = messages_to_prompt(messages, system_prompt="Be helpful.")
        expected = f"{BOS} {B_INST} {_sys_block('Be helpful.')} Hello {E_INST}"
        assert result == expected, (
            f"Single-turn Llama 2 prompt mismatch.\nGot:      {result!r}\n"
            f"Expected: {expected!r}"
        )

    def test_system_message_extracted_from_messages_list(self) -> None:
        """Test that a leading SYSTEM message is used as the system block.

        Test scenario:
            When the first message has role SYSTEM, its content replaces the
            system_prompt argument (which is not passed) in the output.
        """
        messages = [_system("Be helpful."), _user("Hello")]
        result = messages_to_prompt(messages)
        expected = f"{BOS} {B_INST} {_sys_block('Be helpful.')} Hello {E_INST}"
        assert result == expected, (
            f"System-from-messages Llama 2 prompt mismatch.\nGot:      {result!r}\n"
            f"Expected: {expected!r}"
        )

    def test_system_message_whitespace_is_stripped(self) -> None:
        """Test that surrounding whitespace in the SYSTEM message is stripped.

        Test scenario:
            A SYSTEM message with leading/trailing spaces should produce the
            same output as one without whitespace.
        """
        padded = messages_to_prompt([_system("  Be helpful.  "), _user("Hello")])
        clean = messages_to_prompt([_system("Be helpful."), _user("Hello")])
        assert padded == clean, (
            f"Whitespace in system message should be stripped.\n"
            f"Padded:  {padded!r}\nClean: {clean!r}"
        )

    def test_uses_default_system_prompt_when_none_provided(self) -> None:
        """Test that DEFAULT_SYSTEM_PROMPT is used when no system is specified.

        Test scenario:
            No SYSTEM message and no system_prompt argument — the formatted
            system block must embed DEFAULT_SYSTEM_PROMPT (stripped).
        """
        result = messages_to_prompt([_user("Hi")])
        assert DEFAULT_SYSTEM_PROMPT.strip() in result, (
            f"DEFAULT_SYSTEM_PROMPT should appear in output when no system given.\n"
            f"Got: {result!r}"
        )

    def test_explicit_system_prompt_replaces_default(self) -> None:
        """Test that an explicit system_prompt replaces DEFAULT_SYSTEM_PROMPT.

        Test scenario:
            When system_prompt is provided, DEFAULT_SYSTEM_PROMPT must NOT
            appear in the output.
        """
        result = messages_to_prompt([_user("Hi")], system_prompt="Custom.")
        assert DEFAULT_SYSTEM_PROMPT.strip() not in result, (
            f"DEFAULT_SYSTEM_PROMPT must not appear when system_prompt is provided.\n"
            f"Got: {result!r}"
        )
        assert "Custom." in result, (
            f"Explicit system_prompt 'Custom.' should appear in output.\n"
            f"Got: {result!r}"
        )

    def test_multi_turn_user_assistant_user(self) -> None:
        """Test three-message USER/ASSISTANT/USER conversation.

        Test scenario:
            The first turn's [/INST] block is followed by the assistant reply,
            then ``</s>``.  The second USER turn opens a new ``<s> [INST]``
            block without a system block.
        """
        messages = [_user("Q1"), _assist("A1"), _user("Q2")]
        result = messages_to_prompt(messages, system_prompt="sys")
        expected = (
            f"{BOS} {B_INST} {_sys_block('sys')} Q1 {E_INST} A1 {EOS}"
            f"{BOS} {B_INST} Q2 {E_INST}"
        )
        assert result == expected, (
            f"Three-message Llama 2 prompt mismatch.\n"
            f"Got:      {result!r}\nExpected: {expected!r}"
        )

    def test_two_full_turns_user_assist_user_assist(self) -> None:
        """Test four-message USER/ASSISTANT/USER/ASSISTANT conversation.

        Test scenario:
            Both turns should be separated by EOS and the second turn must
            not include the system block.
        """
        messages = [_user("A"), _assist("B"), _user("C"), _assist("D")]
        result = messages_to_prompt(messages, system_prompt="s")
        expected = (
            f"{BOS} {B_INST} {_sys_block('s')} A {E_INST} B {EOS}"
            f"{BOS} {B_INST} C {E_INST} D"
        )
        assert result == expected, (
            f"Four-message Llama 2 prompt mismatch.\n"
            f"Got:      {result!r}\nExpected: {expected!r}"
        )

    def test_raises_for_wrong_role_at_user_position(self) -> None:
        """Test ValueError when ASSISTANT appears where USER is expected.

        Test scenario:
            A single ASSISTANT message (no leading SYSTEM) lands at position 0
            of ``remaining`` and must trigger ValueError.

            Note: a lone SYSTEM message does NOT raise — it is consumed as the
            system block, leaving ``remaining`` empty.
        """
        messages = [Message(role=MessageRole.ASSISTANT, content="wrong")]
        with pytest.raises(ValueError) as exc_info:
            messages_to_prompt(messages)
        assert "Expected a USER message at position 0" in str(exc_info.value), (
            f"Error message should reference position 0, got: {exc_info.value}"
        )

    def test_raises_for_wrong_role_at_assistant_position(self) -> None:
        """Test ValueError when the second message is not ASSISTANT.

        Test scenario:
            A USER/USER sequence (missing ASSISTANT) should raise ValueError
            mentioning position 1 and the unexpected role.
        """
        messages = [_user("Hello"), _user("Also user")]
        with pytest.raises(ValueError) as exc_info:
            messages_to_prompt(messages)
        assert "Expected an ASSISTANT message at position 1" in str(exc_info.value), (
            f"Error message should reference position 1, got: {exc_info.value}"
        )

    def test_error_message_includes_actual_role(self) -> None:
        """Test that the ValueError names the role that was found.

        Test scenario:
            When ASSISTANT appears where USER is expected (position 0),
            the error message must identify the unexpected role so users can
            diagnose the issue without reading source code.
        """
        messages = [Message(role=MessageRole.ASSISTANT, content="wrong")]
        with pytest.raises(ValueError) as exc_info:
            messages_to_prompt(messages)
        error_text = str(exc_info.value)
        assert "ASSISTANT" in error_text or "assistant" in error_text.lower(), (
            f"Error message should name the unexpected role, got: {error_text}"
        )

    def test_bos_token_starts_each_turn(self) -> None:
        """Test that BOS ``<s>`` appears at the start of each conversation turn.

        Test scenario:
            Each new USER turn should begin with BOS to satisfy the Llama 2
            tokeniser expectation for multi-turn contexts.
        """
        messages = [_user("A"), _assist("B"), _user("C")]
        result = messages_to_prompt(messages, system_prompt="s")
        assert result.count(BOS) == 2, (
            f"Expected 2 BOS tokens for a 2-turn conversation, "
            f"found {result.count(BOS)} in {result!r}"
        )

    def test_eos_token_separates_turns(self) -> None:
        """Test that EOS ``</s>`` acts as a turn separator, not a terminator.

        Test scenario:
            EOS is appended only when a subsequent USER turn follows — for 2
            turns (USER/ASSIST/USER/ASSIST), exactly 1 EOS appears between
            them; the final assistant reply has no EOS.
        """
        messages = [_user("A"), _assist("B"), _user("C"), _assist("D")]
        result = messages_to_prompt(messages, system_prompt="s")
        assert result.count(EOS) == 1, (
            f"Expected 1 EOS token between the two turns, "
            f"found {result.count(EOS)} in {result!r}"
        )

    def test_no_eos_when_last_turn_has_no_assistant(self) -> None:
        """Test EOS count for a conversation ending on a USER turn.

        Test scenario:
            A USER/ASSISTANT/USER sequence: the one completed assistant reply
            produces exactly 1 EOS (when processing the next USER turn).
        """
        messages = [_user("A"), _assist("B"), _user("C")]
        result = messages_to_prompt(messages, system_prompt="s")
        assert result.count(EOS) == 1, (
            f"Expected 1 EOS for one completed assistant turn, "
            f"found {result.count(EOS)} in {result!r}"
        )


@pytest.mark.unit
class TestCompletionToPrompt:
    """Tests for completion_to_prompt (Llama 2 completion format)."""

    def test_with_explicit_system_prompt(self) -> None:
        """Test basic completion prompt with an explicit system string.

        Test scenario:
            completion_to_prompt should wrap the completion in a single
            ``<s> [INST] … [/INST]`` block with the system block embedded.
        """
        result = completion_to_prompt("Hello", "Be helpful.")
        expected = f"{BOS} {B_INST} {B_SYS} Be helpful. {E_SYS} Hello {E_INST}"
        assert result == expected, (
            f"completion_to_prompt output mismatch.\n"
            f"Got:      {result!r}\nExpected: {expected!r}"
        )

    def test_uses_default_system_prompt_when_none_provided(self) -> None:
        """Test that DEFAULT_SYSTEM_PROMPT is used when system_prompt is None.

        Test scenario:
            Calling completion_to_prompt without system_prompt should embed
            DEFAULT_SYSTEM_PROMPT (stripped) in the output.
        """
        result = completion_to_prompt("Hello")
        assert DEFAULT_SYSTEM_PROMPT.strip() in result, (
            f"DEFAULT_SYSTEM_PROMPT should appear when system_prompt is None.\n"
            f"Got: {result!r}"
        )

    def test_completion_whitespace_stripped(self) -> None:
        """Test that surrounding whitespace in completion is stripped.

        Test scenario:
            ``  Hello  `` and ``Hello`` should produce identical prompts.
        """
        padded = completion_to_prompt("  Hello  ", "sys")
        clean = completion_to_prompt("Hello", "sys")
        assert padded == clean, (
            f"Completion whitespace should be stripped.\n"
            f"Padded: {padded!r}\nClean:  {clean!r}"
        )

    def test_system_prompt_whitespace_stripped(self) -> None:
        """Test that surrounding whitespace in system_prompt is stripped.

        Test scenario:
            ``  Be helpful.  `` and ``Be helpful.`` as system_prompt should
            produce identical prompts.
        """
        padded = completion_to_prompt("Hi", "  Be helpful.  ")
        clean = completion_to_prompt("Hi", "Be helpful.")
        assert padded == clean, (
            f"System prompt whitespace should be stripped.\n"
            f"Padded: {padded!r}\nClean:  {clean!r}"
        )

    def test_empty_completion_produces_valid_prompt(self) -> None:
        """Test that an empty completion string does not raise.

        Test scenario:
            Passing an empty string as completion should still produce a
            well-formed prompt; the user content section will be empty.
        """
        result = completion_to_prompt("", "sys")
        assert B_INST in result, "BOS+INST header should always be present"
        assert E_INST in result, "INST end token should always be present"

    def test_output_starts_with_bos(self) -> None:
        """Test that the output always begins with the BOS token.

        Test scenario:
            Llama 2 requires ``<s>`` at the very start of every prompt.
        """
        result = completion_to_prompt("Hello", "sys")
        assert result.startswith(BOS), (
            f"Output must start with BOS '{BOS}', got: {result[:20]!r}"
        )

    def test_output_ends_with_e_inst(self) -> None:
        """Test that the output always ends with the E_INST token.

        Test scenario:
            Completion prompts must end with ``[/INST]`` so the model knows
            to start generating the assistant reply.
        """
        result = completion_to_prompt("Hello", "sys")
        assert result.endswith(E_INST), (
            f"Output must end with E_INST '{E_INST}', got last chars: {result[-20:]!r}"
        )

    def test_system_block_is_embedded_in_output(self) -> None:
        """Test that both B_SYS and E_SYS delimiters appear in the output.

        Test scenario:
            The system block tokens (``<<SYS>>`` and ``<</SYS>>``) must
            bracket the system prompt content.
        """
        result = completion_to_prompt("Hello", "custom system")
        assert B_SYS in result, f"B_SYS token '{B_SYS!r}' missing from: {result!r}"
        assert E_SYS in result, f"E_SYS token '{E_SYS!r}' missing from: {result!r}"
        assert "custom system" in result, (
            f"System prompt text should appear in output: {result!r}"
        )
