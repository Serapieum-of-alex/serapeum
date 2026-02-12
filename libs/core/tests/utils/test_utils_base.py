"""Tests for serapeum.core.utils.base utility functions."""

from serapeum.core.utils.base import truncate_text


class TestTruncateText:
    """Test cases for truncate_text utility function."""

    def test_text_shorter_than_max_length(self):
        """Test text shorter than max length returns unchanged.

        Inputs: text length less than max_length.

        Expectation: Function returns the original text unchanged.
        This verifies the early return condition when no truncation is needed.
        """
        assert truncate_text("hello", 10) == "hello"

    def test_text_equal_to_max_length(self):
        """Test text equal to max length returns unchanged.

        Inputs: text length exactly equals max_length.

        Expectation: Function returns the original text unchanged.
        This ensures equality boundary behaves like the shorter case.
        """
        assert truncate_text("12345", 5) == "12345"

    def test_text_longer_than_max_length_adds_ellipsis(self):
        """Test text longer than max length adds ellipsis.

        Inputs: text length greater than max_length.

        Expectation: Function returns a string of length max_length, created as text[:max_length-3] + '...'.
        This checks the truncation and ellipsis logic.
        """
        result = truncate_text("abcdefghij", 8)
        assert result == "abcde" + "..."
        assert len(result) == 8

    def test_small_max_length_edge_case(self):
        """Test small max length edge case.

        Inputs: very small max_length (less than length of '...').

        Expectation: Per current implementation, negative slice may produce a string longer than max_length.
        This test documents current behavior rather than enforcing a specific UX rule.
        """
        # With max_length=2: text[: -1] + '...' -> 'abcde...' for this input
        result = truncate_text("abcdef", 2)
        assert result.endswith("...")
        assert result.startswith("abcde")
