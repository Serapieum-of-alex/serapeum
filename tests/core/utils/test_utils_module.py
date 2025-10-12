import base64
from io import BytesIO
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from serapeum.core.utils.utils import resolve_binary, truncate_text


class TestResolveBinary:
    class TestRawBytes:
        def test_raw_bytes_non_base64_no_encoding(self):
            """
            Inputs:
                raw_bytes set to an arbitrary byte sequence; as_base64=False.
            Expectation:
                Per current implementation, function attempts base64 decoding; if decoding fails it
                returns the original bytes, otherwise it returns the decoded bytes.

            This documents the current behavior of the auto-decode logic.
            """
            raw: bytes = b"\xff\x00\xfeRAW-BYTES\x01\x02"
            bio = resolve_binary(raw_bytes=raw, as_base64=False)
            assert isinstance(bio, BytesIO)
            try:
                expected = base64.b64decode(raw)
            except Exception:
                expected = raw
            assert bio.getvalue() == expected

        def test_raw_bytes_non_base64_with_encoding(self):
            """
            Inputs:
                raw_bytes set to an arbitrary byte sequence; as_base64=True.
            Expectation:
                Per current implementation, function decodes then re-encodes to base64; thus
                base64-decoding the result equals either base64.b64decode(raw_bytes) if decoding succeeds
                or the original raw bytes if decoding fails.

            This documents the decode-then-encode behavior for raw bytes.
            """
            raw: bytes = b"\x00\x10not-base64\x99"
            bio = resolve_binary(raw_bytes=raw, as_base64=True)
            try:
                expected_decoded = base64.b64decode(raw)
            except Exception:
                expected_decoded = raw
            assert base64.b64decode(bio.getvalue()) == expected_decoded

        def test_raw_bytes_is_base64_decoded_when_as_base64_false(self):
            """
            Inputs:
                raw_bytes is a base64-encoded payload; as_base64=False.
            Expectation:
                Function detects base64 and returns the decoded binary data in BytesIO.
                This checks the auto-detection and decoding path for base64 input.
            """
            original = b"hello world"
            raw_b64 = base64.b64encode(original)
            bio = resolve_binary(raw_bytes=raw_b64, as_base64=False)
            assert bio.getvalue() == original

        def test_raw_bytes_is_base64_and_as_base64_true_returns_b64(self):
            """
            Inputs:
                raw_bytes is base64-encoded; as_base64=True.
            Expectation:
                Function decodes then re-encodes and returns base64 bytes, equivalent to normalized input.

            This checks that the return remains base64 when requested.
            """
            original = b"binary-\x00-\xff-data"
            raw_b64 = base64.b64encode(original)
            bio = resolve_binary(raw_bytes=raw_b64, as_base64=True)
            assert base64.b64decode(bio.getvalue()) == original

    class TestPath:
        def test_path_bytes_no_encoding(self, tmp_path: Path):
            """
            Inputs:
                path to a file containing arbitrary bytes; as_base64=False.
            Expectation:
                BytesIO contains the exact bytes read from file.

            This checks the "path" source handling with Path object.
            """
            p = tmp_path / "data.bin"
            data = b"\x01\x02payload\x03"
            p.write_bytes(data)
            bio = resolve_binary(path=p, as_base64=False)
            assert bio.getvalue() == data

        def test_path_bytes_with_encoding_str_path(self, tmp_path: Path):
            """
            Inputs:
                path (as string) to a file; as_base64=True.
            Expectation:
                BytesIO contains base64-encoded content of the file.

            This checks string-path handling and base64 encoding option.
            """
            p = tmp_path / "data2.bin"
            data = b"\x09content-2\x10"
            p.write_bytes(data)
            bio = resolve_binary(path=str(p), as_base64=True)
            assert base64.b64decode(bio.getvalue()) == data

    class TestURL:
        def test_url_data_scheme_base64_no_encoding(self):
            """
            Inputs:
                url is a data: URL with base64 payload; as_base64=False.
            Expectation:
                BytesIO contains decoded binary content.

            This checks parsing and decoding of data URLs with base64 flag.
            """
            data = b"ABC\x00DEF"
            url = f"data:application/octet-stream;base64,{base64.b64encode(data).decode()}"
            bio = resolve_binary(url=url, as_base64=False)
            assert bio.getvalue() == data

        def test_url_data_scheme_base64_with_encoding(self):
            """
            Inputs:
                url is a data: URL with base64 payload; as_base64=True.
            Expectation:
                BytesIO contains base64-encoded version of the decoded payload (i.e., remains base64).

            This checks the as_base64 flag with data URLs.
            """
            data = b"\x00\x01\x02hello"
            url = f"data:application/octet-stream;base64,{base64.b64encode(data).decode()}"
            bio = resolve_binary(url=url, as_base64=True)
            assert base64.b64decode(bio.getvalue()) == data

        def test_url_data_scheme_plain_text(self):
            """
            Inputs:
                url is a data: URL with plain text (no base64); as_base64=False.
            Expectation:
                BytesIO contains the UTF-8 bytes of the text portion as-is.

            This checks the non-base64 data URL branch.
            """
            text = "hello-data"
            url = f"data:text/plain,{text}"
            bio = resolve_binary(url=url, as_base64=False)
            assert bio.getvalue() == text.encode("utf-8")

        def test_url_data_scheme_plain_text_as_base64(self):
            """
            Inputs:
                url is a data: URL with plain text; as_base64=True.
            Expectation:
                BytesIO contains base64-encoded UTF-8 bytes of the text.

            This checks encoding behavior for non-base64 data URLs.
            """
            text = "hi"
            url = f"data:text/plain,{text}"
            bio = resolve_binary(url=url, as_base64=True)
            assert base64.b64decode(bio.getvalue()) == text.encode("utf-8")

        def test_url_data_scheme_invalid_format_raises(self):
            """
            Inputs:
                url is a malformed data: URL missing the comma separator.
            Expectation:
                ValueError is raised indicating invalid format.

            This checks error handling for improperly formatted data URLs.
            """
            bad_url = "data:text/plain;base64SGVsbG8="  # missing comma
            with pytest.raises(ValueError):
                resolve_binary(url=bad_url)

        def test_http_url_fetches_and_respects_as_base64(self):
            """
            Inputs:
                url is an http(s) URL; as_base64 toggles output format.
            Expectation:
                requests.get is called, response.raise_for_status is used, and content is returned
                either raw or base64 encoded depending on the flag.

            This checks the external HTTP fetch path with proper error handling.
            """
            content = b"net-bytes\x00\x01"

            class DummyResponse:
                def __init__(self, content: Optional[bytes]):
                    self.content = content
                    self.raise_called = False

                def raise_for_status(self):
                    self.raise_called = True

            dummy = DummyResponse(content)
            with patch("serapeum.core.utils.utils.requests.get", return_value=dummy) as mock_get:
                # as_base64=False
                bio = resolve_binary(url="https://example.com/data.bin", as_base64=False)
                assert bio.getvalue() == content
                assert dummy.raise_called is True
                mock_get.assert_called_once()

            dummy2 = DummyResponse(content)
            with patch("serapeum.core.utils.utils.requests.get", return_value=dummy2):
                # as_base64=True
                bio2 = resolve_binary(url="https://example.com/data.bin", as_base64=True)
                assert base64.b64decode(bio2.getvalue()) == content

        def test_no_valid_source_raises(self):
            """Inputs: All source arguments are None (no raw_bytes, no path, no url).
            Expectation: ValueError is raised indicating no valid source was provided.
            This checks the final error path.
            """
            with pytest.raises(ValueError):
                resolve_binary()


class TestTruncateText:
    def test_text_shorter_than_max_length(self):
        """Inputs: text length less than max_length.
        Expectation: Function returns the original text unchanged.
        This verifies the early return condition when no truncation is needed.
        """
        assert truncate_text("hello", 10) == "hello"

    def test_text_equal_to_max_length(self):
        """Inputs: text length exactly equals max_length.
        Expectation: Function returns the original text unchanged.
        This ensures equality boundary behaves like the shorter case.
        """
        assert truncate_text("12345", 5) == "12345"

    def test_text_longer_than_max_length_adds_ellipsis(self):
        """Inputs: text length greater than max_length.
        Expectation: Function returns a string of length max_length, created as text[:max_length-3] + '...'.
        This checks the truncation and ellipsis logic.
        """
        result = truncate_text("abcdefghij", 8)
        assert result == "abcde" + "..."
        assert len(result) == 8

    def test_small_max_length_edge_case(self):
        """Inputs: very small max_length (less than length of '...').
        Expectation: Per current implementation, negative slice may produce a string longer than max_length.
        This test documents current behavior rather than enforcing a specific UX rule.
        """
        # With max_length=2: text[: -1] + '...' -> 'abcde...' for this input
        result = truncate_text("abcdef", 2)
        assert result.endswith("...")
        assert result.startswith("abcde")
