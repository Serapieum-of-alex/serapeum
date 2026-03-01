"""Unit tests for serapeum.llama_cpp.formatters.llama3 and formatters.__init__.

Note: test_llms_llama_cpp.py already covers:
  - completion_to_prompt_v3_instruct("USER MESSAGE", "SYSTEM PROMPT") — exact output
  - messages_to_prompt_v3_instruct with a 3-message USER/ASSISTANT/USER list

The tests below cover complementary scenarios to reach full branch coverage.
"""

from __future__ import annotations

import pytest

from serapeum.core.llms import Message, MessageRole
from serapeum.llama_cpp.formatters.llama3 import (
    DEFAULT_SYSTEM_PROMPT,
    EOT,
    HEADER_ASSIST,
    HEADER_SYS,
    HEADER_USER,
    completion_to_prompt_v3_instruct,
    messages_to_prompt_v3_instruct,
)


def _user(content: str) -> Message:
    return Message(role=MessageRole.USER, content=content)


def _assist(content: str) -> Message:
    return Message(role=MessageRole.ASSISTANT, content=content)


def _system(content: str) -> Message:
    return Message(role=MessageRole.SYSTEM, content=content)


@pytest.mark.unit
class TestMessagesToPromptV3Instruct:
    """Tests for messages_to_prompt_v3_instruct (Llama 3 Instruct format)."""

    def test_single_user_message_with_system_prompt(self) -> None:
        """Test basic single-turn prompt with an explicit system string.

        Test scenario:
            A single USER message and an explicit system_prompt should produce
            a system header block, a user block, and a trailing assistant header.
        """
        messages = [_user("Hello")]
        result = messages_to_prompt_v3_instruct(messages, system_prompt="sys")
        expected = f"{HEADER_SYS}sys{EOT}" f"{HEADER_USER}Hello{EOT}" f"{HEADER_ASSIST}"
        assert result == expected, (
            f"Single-turn Llama 3 prompt mismatch.\n"
            f"Got:      {result!r}\nExpected: {expected!r}"
        )

    def test_system_message_extracted_from_first_message(self) -> None:
        """Test that a leading SYSTEM message is used as the system block.

        Test scenario:
            When the first message has role SYSTEM, its content fills the
            system header and the remaining messages form the conversation.
        """
        messages = [_system("sys"), _user("Hello")]
        result = messages_to_prompt_v3_instruct(messages)
        expected = f"{HEADER_SYS}sys{EOT}" f"{HEADER_USER}Hello{EOT}" f"{HEADER_ASSIST}"
        assert result == expected, (
            f"System-from-messages Llama 3 prompt mismatch.\n"
            f"Got:      {result!r}\nExpected: {expected!r}"
        )

    def test_system_message_whitespace_stripped(self) -> None:
        """Test that surrounding whitespace in the SYSTEM message is stripped.

        Test scenario:
            A SYSTEM message of ``  sys  `` should produce the same system
            header as one containing ``sys``.
        """
        padded = messages_to_prompt_v3_instruct([_system("  sys  "), _user("Hello")])
        clean = messages_to_prompt_v3_instruct([_system("sys"), _user("Hello")])
        assert padded == clean, (
            f"Whitespace in system message should be stripped.\n"
            f"Padded: {padded!r}\nClean:  {clean!r}"
        )

    def test_uses_default_system_prompt_when_none_provided(self) -> None:
        """Test that DEFAULT_SYSTEM_PROMPT is used when no system is given.

        Test scenario:
            No SYSTEM message and no system_prompt argument — the formatted
            system header must embed DEFAULT_SYSTEM_PROMPT (stripped).
        """
        result = messages_to_prompt_v3_instruct([_user("Hi")])
        assert DEFAULT_SYSTEM_PROMPT.strip() in result, (
            f"DEFAULT_SYSTEM_PROMPT should appear when no system is provided.\n"
            f"Got: {result!r}"
        )

    def test_explicit_system_prompt_replaces_default(self) -> None:
        """Test that an explicit system_prompt replaces DEFAULT_SYSTEM_PROMPT.

        Test scenario:
            When system_prompt is supplied, DEFAULT_SYSTEM_PROMPT must not
            appear anywhere in the output.
        """
        result = messages_to_prompt_v3_instruct([_user("Hi")], system_prompt="Custom.")
        assert DEFAULT_SYSTEM_PROMPT.strip() not in result, (
            f"DEFAULT_SYSTEM_PROMPT must not appear when system_prompt is given.\n"
            f"Got: {result!r}"
        )
        assert (
            "Custom." in result
        ), f"Custom system prompt should appear in output.\nGot: {result!r}"

    def test_multi_turn_user_assistant_user(self) -> None:
        """Test three-message USER/ASSISTANT/USER conversation structure.

        Test scenario:
            The first assistant reply is embedded in the first turn block,
            and the second user message opens a new user block with only the
            trailing HEADER_ASSIST following it.
        """
        messages = [_user("A"), _assist("B"), _user("C")]
        result = messages_to_prompt_v3_instruct(messages, system_prompt="sys")
        expected = (
            f"{HEADER_SYS}sys{EOT}"
            f"{HEADER_USER}A{EOT}"
            f"{HEADER_ASSIST}B{EOT}"
            f"{HEADER_USER}C{EOT}"
            f"{HEADER_ASSIST}"
        )
        assert result == expected, (
            f"Three-message Llama 3 prompt mismatch.\n"
            f"Got:      {result!r}\nExpected: {expected!r}"
        )

    def test_two_full_turns_user_assist_user_assist(self) -> None:
        """Test four-message USER/ASSISTANT/USER/ASSISTANT conversation.

        Test scenario:
            Both assistant replies should appear immediately after their
            respective user turns, each terminated with EOT.
        """
        messages = [_user("A"), _assist("B"), _user("C"), _assist("D")]
        result = messages_to_prompt_v3_instruct(messages, system_prompt="sys")
        expected = (
            f"{HEADER_SYS}sys{EOT}"
            f"{HEADER_USER}A{EOT}"
            f"{HEADER_ASSIST}B{EOT}"
            f"{HEADER_USER}C{EOT}"
            f"{HEADER_ASSIST}D{EOT}"
            f"{HEADER_ASSIST}"
        )
        assert result == expected, (
            f"Four-message Llama 3 prompt mismatch.\n"
            f"Got:      {result!r}\nExpected: {expected!r}"
        )

    def test_always_ends_with_assistant_header(self) -> None:
        """Test that the output invariably ends with HEADER_ASSIST.

        Test scenario:
            Regardless of message count or content, the prompt must always end
            with the bare assistant header to cue the model to start generating.
        """
        for msgs in (
            [_user("A")],
            [_user("A"), _assist("B"), _user("C")],
        ):
            result = messages_to_prompt_v3_instruct(msgs, system_prompt="s")
            assert result.endswith(
                HEADER_ASSIST
            ), f"Prompt must end with HEADER_ASSIST.\nGot: {result!r}"

    def test_raises_for_wrong_role_at_user_position(self) -> None:
        """Test ValueError when ASSISTANT appears where USER is expected.

        Test scenario:
            A single ASSISTANT message (no leading SYSTEM) lands at position 0
            of ``remaining`` and must trigger ValueError.

            Note: a lone SYSTEM message does NOT raise — it is consumed as the
            system block, leaving ``remaining`` empty (the trailing
            HEADER_ASSIST is still appended).
        """
        messages = [Message(role=MessageRole.ASSISTANT, content="wrong")]
        with pytest.raises(ValueError) as exc_info:
            messages_to_prompt_v3_instruct(messages)
        assert "Expected a USER message at position 0" in str(
            exc_info.value
        ), f"Error should mention position 0, got: {exc_info.value}"

    def test_raises_for_wrong_role_at_assistant_position(self) -> None:
        """Test ValueError when the second message is not ASSISTANT.

        Test scenario:
            USER/USER is an invalid alternation pattern — ValueError must be
            raised mentioning position 1.
        """
        messages = [_user("Hello"), _user("Also user")]
        with pytest.raises(ValueError) as exc_info:
            messages_to_prompt_v3_instruct(messages)
        assert "Expected an ASSISTANT message at position 1" in str(
            exc_info.value
        ), f"Error should mention position 1, got: {exc_info.value}"

    def test_error_message_includes_actual_role(self) -> None:
        """Test that the ValueError names the role that was found.

        Test scenario:
            When ASSISTANT appears where USER is expected at position 0,
            the error message must identify the unexpected role.
        """
        messages = [Message(role=MessageRole.ASSISTANT, content="wrong")]
        with pytest.raises(ValueError) as exc_info:
            messages_to_prompt_v3_instruct(messages)
        error = str(exc_info.value)
        assert (
            "ASSISTANT" in error or "assistant" in error.lower()
        ), f"Error should name the unexpected role, got: {error}"

    def test_eot_appears_after_every_role_block(self) -> None:
        """Test that EOT terminates every role block in the output.

        Test scenario:
            System, each user turn, and each assistant reply must all carry
            EOT; the trailing bare HEADER_ASSIST has none.
            sys(1) + user A(1) + assist B(1) + user C(1) = 4 EOT tokens.
        """
        messages = [_user("A"), _assist("B"), _user("C")]
        result = messages_to_prompt_v3_instruct(messages, system_prompt="s")
        assert (
            result.count(EOT) == 4
        ), f"Expected 4 EOT tokens for sys+u+a+u, found {result.count(EOT)} in {result!r}"


@pytest.mark.unit
class TestCompletionToPromptV3Instruct:
    """Tests for completion_to_prompt_v3_instruct (Llama 3 completion format).

    Note: the scenario (completion="USER MESSAGE", system_prompt="SYSTEM PROMPT")
    is already tested in test_llms_llama_cpp.py::test_completion_to_prompt_v3_instruct.
    """

    def test_completion_whitespace_stripped(self) -> None:
        """Test that surrounding whitespace in completion is stripped.

        Test scenario:
            ``  Hello  `` and ``Hello`` should produce identical prompts.
        """
        padded = completion_to_prompt_v3_instruct("  Hello  ", "sys")
        clean = completion_to_prompt_v3_instruct("Hello", "sys")
        assert padded == clean, (
            f"Completion whitespace should be stripped.\n"
            f"Padded: {padded!r}\nClean:  {clean!r}"
        )

    def test_system_prompt_whitespace_stripped(self) -> None:
        """Test that surrounding whitespace in system_prompt is stripped.

        Test scenario:
            ``  sys  `` and ``sys`` as system_prompt should produce identical prompts.
        """
        padded = completion_to_prompt_v3_instruct("Hello", "  sys  ")
        clean = completion_to_prompt_v3_instruct("Hello", "sys")
        assert padded == clean, (
            f"System prompt whitespace should be stripped.\n"
            f"Padded: {padded!r}\nClean:  {clean!r}"
        )

    def test_empty_completion_produces_valid_prompt(self) -> None:
        """Test that an empty completion string produces a structurally valid prompt.

        Test scenario:
            An empty string completion should produce a prompt with an empty
            user block (``HEADER_USER + EOT``) rather than raise.
        """
        result = completion_to_prompt_v3_instruct("", "sys")
        expected = f"{HEADER_SYS}sys{EOT}" f"{HEADER_USER}{EOT}" f"{HEADER_ASSIST}"
        assert result == expected, (
            f"Empty completion prompt mismatch.\n"
            f"Got:      {result!r}\nExpected: {expected!r}"
        )

    def test_uses_default_system_prompt_when_none_provided(self) -> None:
        """Test that DEFAULT_SYSTEM_PROMPT is embedded when system_prompt is None.

        Test scenario:
            Calling without system_prompt should include DEFAULT_SYSTEM_PROMPT
            (stripped) in the system header block.
        """
        result = completion_to_prompt_v3_instruct("Hello")
        assert DEFAULT_SYSTEM_PROMPT.strip() in result, (
            f"DEFAULT_SYSTEM_PROMPT should appear when system_prompt is None.\n"
            f"Got: {result!r}"
        )

    def test_always_ends_with_assistant_header(self) -> None:
        """Test that every output ends with HEADER_ASSIST.

        Test scenario:
            The assistant header must always be the final token so the model
            knows to continue from the assistant's perspective.
        """
        result = completion_to_prompt_v3_instruct("Hello", "sys")
        assert result.endswith(
            HEADER_ASSIST
        ), f"Output must end with HEADER_ASSIST.\nGot: {result!r}"

    def test_output_structure_is_sys_user_assist(self) -> None:
        """Test the canonical three-block structure: system → user → assistant.

        Test scenario:
            The output must contain HEADER_SYS before HEADER_USER, which must
            come before HEADER_ASSIST — in that exact order.
        """
        result = completion_to_prompt_v3_instruct("Hi", "s")
        idx_sys = result.index(HEADER_SYS)
        idx_user = result.index(HEADER_USER)
        idx_assist = result.index(HEADER_ASSIST)
        assert idx_sys < idx_user < idx_assist, (
            f"Expected header order sys < user < assist.\n"
            f"Positions: sys={idx_sys}, user={idx_user}, assist={idx_assist}"
        )

    def test_exact_output_with_custom_values(self) -> None:
        """Test exact output for a known input pair.

        Test scenario:
            Fixed inputs ``("Prompt", "Instruction")`` must produce a
            deterministic byte-for-byte output.
        """
        result = completion_to_prompt_v3_instruct("Prompt", "Instruction")
        expected = (
            f"{HEADER_SYS}Instruction{EOT}"
            f"{HEADER_USER}Prompt{EOT}"
            f"{HEADER_ASSIST}"
        )
        assert (
            result == expected
        ), f"Exact output mismatch.\nGot:      {result!r}\nExpected: {expected!r}"


@pytest.mark.unit
class TestFormattersPackageExports:
    """Tests that serapeum.llama_cpp.formatters exposes llama2 and llama3 submodules."""

    def test_llama2_submodule_importable(self) -> None:
        """Test llama2 submodule is importable from the formatters package.

        Test scenario:
            ``from serapeum.llama_cpp.formatters import llama2`` must succeed
            and be the same module as ``serapeum.llama_cpp.formatters.llama2``.
        """
        import serapeum.llama_cpp.formatters.llama2 as direct_mod
        from serapeum.llama_cpp.formatters import llama2 as pkg_mod

        assert pkg_mod is direct_mod

    def test_llama3_submodule_importable(self) -> None:
        """Test llama3 submodule is importable from the formatters package.

        Test scenario:
            ``from serapeum.llama_cpp.formatters import llama3`` must succeed
            and be the same module as ``serapeum.llama_cpp.formatters.llama3``.
        """
        import serapeum.llama_cpp.formatters.llama3 as direct_mod
        from serapeum.llama_cpp.formatters import llama3 as pkg_mod

        assert pkg_mod is direct_mod

    def test_llama2_functions_accessible(self) -> None:
        """Test llama2 formatter functions are accessible via the submodule.

        Test scenario:
            ``llama2.messages_to_prompt`` and ``llama2.completion_to_prompt``
            must be callable.
        """
        from serapeum.llama_cpp.formatters import llama2

        assert callable(llama2.messages_to_prompt)
        assert callable(llama2.completion_to_prompt)

    def test_llama3_functions_accessible(self) -> None:
        """Test llama3 formatter functions are accessible via the submodule.

        Test scenario:
            ``llama3.messages_to_prompt_v3_instruct`` and
            ``llama3.completion_to_prompt_v3_instruct`` must be callable.
        """
        from serapeum.llama_cpp.formatters import llama3

        assert callable(llama3.messages_to_prompt_v3_instruct)
        assert callable(llama3.completion_to_prompt_v3_instruct)

    def test_all_defines_submodule_names(self) -> None:
        """Test that __all__ exposes exactly the two submodule names.

        Test scenario:
            The package's ``__all__`` list must contain ``llama2`` and
            ``llama3`` and nothing else.
        """
        import serapeum.llama_cpp.formatters as pkg

        expected = {"llama2", "llama3"}
        assert set(pkg.__all__) == expected, (
            f"formatters.__all__ mismatch.\n"
            f"Got:      {set(pkg.__all__)}\nExpected: {expected}"
        )
