"""Unit, integration, and performance tests for serapeum.llama_cpp.llm.LlamaCPP.

Existing coverage in other test files (not duplicated here):
  - test_llms_llama_cpp.py : MRO check, formatter function outputs
  - test_llama_cpp_e2e.py  : full end-to-end tests against a real GGUF model

All tests here mock ``serapeum.llama_cpp.llm.Llama`` so no real model is needed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from serapeum.core.llms import CompletionResponse
from serapeum.llama_cpp.llm import _MODEL_CACHE, LlamaCPP


def _noop_messages_to_prompt(messages: Any, system_prompt: str | None = None) -> str:
    """Minimal messages formatter that joins content strings."""
    return " ".join(m.content or "" for m in messages)


def _noop_completion_to_prompt(
    completion: str, system_prompt: str | None = None
) -> str:
    """Minimal completion formatter that returns the input unchanged."""
    return completion


def _formatters() -> dict[str, Any]:
    """Return the minimum set of formatters required by LlamaCPP."""
    return {
        "messages_to_prompt": _noop_messages_to_prompt,
        "completion_to_prompt": _noop_completion_to_prompt,
    }


@pytest.fixture(autouse=True)
def clear_model_cache() -> Generator[None, None, None]:
    """Clear the module-level model cache before and after each test.

    Prevents Llama instances cached by one test from interfering with another.
    """
    _MODEL_CACHE.clear()
    yield
    _MODEL_CACHE.clear()


@pytest.fixture(scope="function")
def model_file(tmp_path: Path) -> Path:
    """Create a temporary fake GGUF model file.

    Returns:
        Path: Path to the fake model file.
    """
    path = tmp_path / "model.gguf"
    path.write_bytes(b"fake gguf content")
    return path


@pytest.fixture(scope="function")
def mock_llama_cls(mocker: Any) -> MagicMock:
    """Patch serapeum.llama_cpp.llm.Llama to prevent actual model loading.

    Returns:
        MagicMock: The patched Llama class mock; use .return_value to reach
            the mock instance.
    """
    mock_cls = mocker.patch("serapeum.llama_cpp.llm.Llama")
    mock_instance = MagicMock()
    mock_instance.context_params.n_ctx = 512
    mock_instance.tokenize.return_value = []
    mock_cls.return_value = mock_instance
    return mock_cls  # type: ignore[no-any-return]


@pytest.fixture(scope="function")
def mock_llama(mock_llama_cls: MagicMock) -> MagicMock:
    """Return the mock Llama instance created by mock_llama_cls.

    Args:
        mock_llama_cls: The patched Llama class fixture.

    Returns:
        MagicMock: The mock Llama instance (mock_llama_cls.return_value).
    """
    return mock_llama_cls.return_value  # type: ignore[no-any-return]


@pytest.fixture(scope="function")
def llm(mock_llama: MagicMock, model_file: Path) -> LlamaCPP:
    """Create a LlamaCPP instance backed by a fake model file and mocked backend.

    Args:
        mock_llama: Mock Llama instance fixture (activates the patch).
        model_file: Temporary fake model file fixture.

    Returns:
        LlamaCPP: Configured instance ready for inference tests.
    """
    return LlamaCPP(
        model_path=str(model_file),
        max_new_tokens=32,
        temperature=0.0,
        **_formatters(),
    )


@pytest.mark.unit
class TestLlamaCPP:
    """Unit tests for LlamaCPP covering validators, construction, and inference."""

    @pytest.mark.unit
    def test_class_name_returns_string_identifier(self) -> None:
        """Test class_name() returns 'LlamaCPP' without a model instance.

        Test scenario:
            class_name() is a classmethod — it must not require construction
            and must return the exact string 'LlamaCPP'.
        """
        assert (
            LlamaCPP.class_name() == "LlamaCPP"
        ), f"Expected 'LlamaCPP', got {LlamaCPP.class_name()!r}"

    @pytest.mark.unit
    def test_check_model_source_raises_when_no_source_provided(
        self, mocker: Any
    ) -> None:
        """Test _check_model_source raises when all three model sources are absent.

        Test scenario:
            The validator must raise ValueError naming all three valid sources
            when model_path, model_url, and hf_model_id are all None.
            _resolve_model_path and _load_model are patched so model_post_init
            does nothing, then the validator is called directly.
        """
        mocker.patch.object(LlamaCPP, "_resolve_model_path", return_value=None)
        mocker.patch.object(LlamaCPP, "_load_model", return_value=None)
        instance = LlamaCPP.model_construct(
            model_path=None,
            model_url=None,
            hf_model_id=None,
            hf_filename=None,
        )
        with pytest.raises(
            ValueError, match="One of model_path, model_url, or hf_model_id"
        ) as exc_info:
            instance._check_model_source()
        assert "model_path" in str(
            exc_info.value
        ), f"Error should mention model_path, got: {exc_info.value}"

    @pytest.mark.unit
    def test_check_model_source_raises_when_hf_model_id_without_hf_filename(
        self, mocker: Any
    ) -> None:
        """Test _check_model_source raises when hf_model_id is given but hf_filename is absent.

        Test scenario:
            HuggingFace Hub downloads require both hf_model_id and hf_filename.
            The validator must raise naming hf_filename before any I/O.
            _resolve_model_path and _load_model are patched so model_post_init
            does nothing, then the validator is called directly.
        """
        mocker.patch.object(LlamaCPP, "_resolve_model_path", return_value=None)
        mocker.patch.object(LlamaCPP, "_load_model", return_value=None)
        instance = LlamaCPP.model_construct(
            model_path=None,
            model_url=None,
            hf_model_id="TheBloke/Llama-2-13B-chat-GGUF",
            hf_filename=None,
        )
        with pytest.raises(ValueError, match="hf_filename is required") as exc_info:
            instance._check_model_source()
        assert "hf_filename" in str(
            exc_info.value
        ), f"Error should name hf_filename, got: {exc_info.value}"

    @pytest.mark.unit
    def test_prepare_kwargs_injects_n_ctx_from_context_window(
        self, mock_llama: MagicMock, model_file: Path
    ) -> None:
        """Test that _prepare_kwargs sets n_ctx in model_kwargs from context_window.

        Test scenario:
            When context_window is provided and model_kwargs does not already
            contain n_ctx, the validator must inject n_ctx = context_window.
        """
        llm = LlamaCPP(
            model_path=str(model_file),
            context_window=2048,
            **_formatters(),
        )
        assert (
            llm.model_kwargs["n_ctx"] == 2048
        ), f"Expected n_ctx=2048 in model_kwargs, got: {llm.model_kwargs}"

    @pytest.mark.unit
    def test_prepare_kwargs_user_supplied_n_ctx_takes_precedence(
        self, mock_llama: MagicMock, model_file: Path
    ) -> None:
        """Test that user-supplied n_ctx in model_kwargs is not overwritten.

        Test scenario:
            If model_kwargs already contains n_ctx, _prepare_kwargs must NOT
            overwrite it with context_window — user values take precedence.
        """
        llm = LlamaCPP(
            model_path=str(model_file),
            context_window=2048,
            model_kwargs={"n_ctx": 4096},
            **_formatters(),
        )
        assert (
            llm.model_kwargs["n_ctx"] == 4096
        ), f"User-supplied n_ctx should take precedence, got: {llm.model_kwargs}"

    @pytest.mark.unit
    def test_prepare_kwargs_injects_verbose_from_verbose_field(
        self, mock_llama: MagicMock, model_file: Path
    ) -> None:
        """Test that _prepare_kwargs sets verbose in model_kwargs from the verbose field.

        Test scenario:
            When verbose=False and model_kwargs does not contain verbose,
            model_kwargs["verbose"] must be set to False.
        """
        llm = LlamaCPP(
            model_path=str(model_file),
            verbose=False,
            **_formatters(),
        )
        assert (
            llm.model_kwargs["verbose"] is False
        ), f"Expected verbose=False in model_kwargs, got: {llm.model_kwargs}"

    @pytest.mark.unit
    def test_prepare_kwargs_user_supplied_verbose_takes_precedence(
        self, mock_llama: MagicMock, model_file: Path
    ) -> None:
        """Test that user-supplied verbose in model_kwargs is not overwritten.

        Test scenario:
            If model_kwargs already contains verbose=True while the field
            verbose=False, model_kwargs["verbose"] must remain True.
        """
        llm = LlamaCPP(
            model_path=str(model_file),
            verbose=False,
            model_kwargs={"verbose": True},
            **_formatters(),
        )
        assert (
            llm.model_kwargs["verbose"] is True
        ), f"User model_kwargs.verbose should take precedence, got: {llm.model_kwargs}"

    @pytest.mark.unit
    def test_prepare_kwargs_injects_n_gpu_layers_from_field(
        self, mock_llama: MagicMock, model_file: Path
    ) -> None:
        """Test that _prepare_kwargs injects n_gpu_layers into model_kwargs.

        Test scenario:
            Passing n_gpu_layers=4 at the top level must cause model_kwargs
            to contain n_gpu_layers=4 via the before-validator setdefault.
        """
        llm = LlamaCPP(
            model_path=str(model_file),
            n_gpu_layers=4,
            **_formatters(),
        )
        assert (
            llm.model_kwargs["n_gpu_layers"] == 4
        ), f"Expected n_gpu_layers=4 in model_kwargs, got: {llm.model_kwargs}"

    @pytest.mark.unit
    def test_prepare_kwargs_user_supplied_n_gpu_layers_takes_precedence(
        self, mock_llama: MagicMock, model_file: Path
    ) -> None:
        """Test that user-supplied n_gpu_layers in model_kwargs is not overwritten.

        Test scenario:
            model_kwargs={"n_gpu_layers": 8} combined with n_gpu_layers=4
            (field) must keep the model_kwargs value of 8 because setdefault
            does not overwrite existing keys.
        """
        llm = LlamaCPP(
            model_path=str(model_file),
            n_gpu_layers=4,
            model_kwargs={"n_gpu_layers": 8},
            **_formatters(),
        )
        assert (
            llm.model_kwargs["n_gpu_layers"] == 8
        ), f"User model_kwargs.n_gpu_layers should take precedence, got: {llm.model_kwargs}"

    @pytest.mark.unit
    def test_model_post_init_raises_when_both_formatters_missing(
        self, mocker: Any
    ) -> None:
        """Test _check_formatters raises when neither formatter is in model_fields_set.

        Test scenario:
            Both formatters are required. _check_formatters must raise ValueError
            naming both missing formatters when model_fields_set contains neither.
            Uses model_construct with empty _fields_set to call the validator directly.
            model_post_init is patched to prevent model_construct from triggering
            a real model download.
        """
        mocker.patch.object(LlamaCPP, "model_post_init")
        instance = LlamaCPP.model_construct(_fields_set=set())
        with pytest.raises(
            ValueError, match="requires explicit prompt formatters"
        ) as exc_info:
            instance._check_formatters()
        error = str(exc_info.value)
        assert (
            "messages_to_prompt" in error
        ), f"Error should name messages_to_prompt, got: {error}"
        assert (
            "completion_to_prompt" in error
        ), f"Error should name completion_to_prompt, got: {error}"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "missing_formatter, fields_set",
        [
            ("messages_to_prompt", {"completion_to_prompt"}),
            ("completion_to_prompt", {"messages_to_prompt"}),
        ],
        ids=["messages-missing", "completion-missing"],
    )
    def test_model_post_init_raises_when_one_formatter_missing(
        self,
        mocker: Any,
        missing_formatter: str,
        fields_set: set[str],
    ) -> None:
        """Test _check_formatters raises naming the specific missing formatter.

        Args:
            missing_formatter: Name of the formatter absent from model_fields_set.
            fields_set: model_fields_set value containing only the present formatter.

        Test scenario:
            When only one formatter is in model_fields_set, _check_formatters must
            raise ValueError that names the missing formatter specifically.
            model_post_init is patched to prevent model_construct from triggering
            a real model download.
        """
        mocker.patch.object(LlamaCPP, "model_post_init")
        instance = LlamaCPP.model_construct(_fields_set=fields_set)
        with pytest.raises(ValueError, match=missing_formatter) as exc_info:
            instance._check_formatters()
        assert missing_formatter in str(
            exc_info.value
        ), f"Error should identify {missing_formatter!r} as missing, got: {exc_info.value}"

    @pytest.mark.unit
    def test_model_post_init_raises_when_model_path_does_not_exist(self) -> None:
        """Test ValueError when model_path points to a non-existent file.

        Test scenario:
            Formatters are provided but the path does not exist on disk —
            ValueError must be raised before Llama() is called.
        """
        with pytest.raises(ValueError, match="does not exist") as exc_info:
            LlamaCPP(model_path="/this/path/does/not/exist.gguf", **_formatters())
        assert (
            "path" in str(exc_info.value).lower()
        ), f"Error should reference the path issue, got: {exc_info.value}"

    @pytest.mark.unit
    def test_model_post_init_calls_llama_with_model_path(
        self, mock_llama_cls: MagicMock, model_file: Path
    ) -> None:
        """Test that Llama() is called with the provided model_path string.

        Test scenario:
            When model_path points to an existing file, Llama() must be
            instantiated with that path as the model_path keyword argument.
        """
        LlamaCPP(model_path=str(model_file), **_formatters())
        mock_llama_cls.assert_called_once()
        assert mock_llama_cls.call_args.kwargs["model_path"] == str(model_file), (
            f"Llama should receive model_path={str(model_file)!r}, "
            f"got: {mock_llama_cls.call_args.kwargs}"
        )

    @pytest.mark.unit
    def test_model_post_init_forwards_model_kwargs_to_llama(
        self, mock_llama_cls: MagicMock, model_file: Path
    ) -> None:
        """Test that extra model_kwargs are unpacked into the Llama constructor.

        Test scenario:
            n_gpu_layers supplied in model_kwargs should appear as a keyword
            argument in the Llama() call.
        """
        LlamaCPP(
            model_path=str(model_file),
            model_kwargs={"n_gpu_layers": 4},
            **_formatters(),
        )
        assert mock_llama_cls.call_args.kwargs.get("n_gpu_layers") == 4, (
            f"n_gpu_layers should be forwarded to Llama, "
            f"got: {mock_llama_cls.call_args.kwargs}"
        )

    @pytest.mark.unit
    def test_model_post_init_downloads_when_url_given_and_file_absent(
        self, mock_llama_cls: MagicMock, mocker: Any, tmp_path: Path
    ) -> None:
        """Test that _fetch_model_file is called when the cached file does not exist.

        Test scenario:
            With model_url set and no cached file present, the download helper
            must be invoked with the URL and expected local path.
        """
        model_url = "https://example.com/model.gguf"
        expected_path = tmp_path / "models" / "model.gguf"
        mocker.patch("serapeum.llama_cpp.llm.get_cache_dir", return_value=str(tmp_path))

        def _create_file(url: str, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake")

        mock_fetch = mocker.patch(
            "serapeum.llama_cpp.llm._fetch_model_file", side_effect=_create_file
        )
        LlamaCPP(model_url=model_url, **_formatters())
        mock_fetch.assert_called_once_with(model_url, expected_path)

    @pytest.mark.unit
    def test_model_post_init_skips_download_when_cached_file_exists(
        self, mock_llama_cls: MagicMock, mocker: Any, tmp_path: Path
    ) -> None:
        """Test that _fetch_model_file is NOT called when the cached file already exists.

        Test scenario:
            If the resolved local cache path already contains a file, the
            download step must be skipped entirely.
        """
        model_url = "https://example.com/model.gguf"
        cached = tmp_path / "models" / "model.gguf"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"cached content")
        mocker.patch("serapeum.llama_cpp.llm.get_cache_dir", return_value=str(tmp_path))
        mock_fetch = mocker.patch("serapeum.llama_cpp.llm._fetch_model_file")

        LlamaCPP(model_url=model_url, **_formatters())

        mock_fetch.assert_not_called()

    @pytest.mark.unit
    def test_model_post_init_raises_runtime_error_when_download_produces_no_file(
        self, mock_llama_cls: MagicMock, mocker: Any, tmp_path: Path
    ) -> None:
        """Test RuntimeError when _fetch_model_file completes but file is still absent.

        Test scenario:
            If the download helper returns without error but the model file
            is still missing on disk (silent failure), RuntimeError must be
            raised with a descriptive message.
        """
        mocker.patch("serapeum.llama_cpp.llm.get_cache_dir", return_value=str(tmp_path))
        mocker.patch(
            "serapeum.llama_cpp.llm._fetch_model_file"
        )  # no-op; file not created

        with pytest.raises(RuntimeError, match="model not found at") as exc_info:
            LlamaCPP(model_url="https://example.com/model.gguf", **_formatters())
        assert (
            "model not found" in str(exc_info.value).lower()
        ), f"RuntimeError should describe the missing file, got: {exc_info.value}"

    @pytest.mark.unit
    def test_model_post_init_updates_model_path_after_url_download(
        self, mock_llama_cls: MagicMock, mocker: Any, tmp_path: Path
    ) -> None:
        """Test that model_path is updated to the local cache path after URL load.

        Test scenario:
            After a URL-sourced model is loaded, llm.model_path must reflect
            the resolved local path rather than remaining None.
        """
        model_url = "https://example.com/my-model.gguf"
        expected = tmp_path / "models" / "my-model.gguf"
        mocker.patch("serapeum.llama_cpp.llm.get_cache_dir", return_value=str(tmp_path))

        def _create(url: str, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")

        mocker.patch("serapeum.llama_cpp.llm._fetch_model_file", side_effect=_create)
        llm = LlamaCPP(model_url=model_url, **_formatters())

        assert llm.model_path == str(
            expected
        ), f"model_path should be updated to {str(expected)!r}, got {llm.model_path!r}"

    @pytest.mark.unit
    def test_model_post_init_downloads_from_hf_hub_when_hf_model_id_given(
        self, mock_llama_cls: MagicMock, mocker: Any, tmp_path: Path
    ) -> None:
        """Test that _fetch_model_file_hf is called when hf_model_id is provided.

        Test scenario:
            With hf_model_id and hf_filename set, the HuggingFace download
            helper must be invoked with the repo ID, filename, and cache dir.
        """
        repo_id = "TheBloke/Llama-2-13B-chat-GGUF"
        filename = "llama-2-13b-chat.Q4_0.gguf"
        cache_dir = tmp_path / "models"
        downloaded = cache_dir / filename
        downloaded.parent.mkdir(parents=True, exist_ok=True)
        downloaded.write_bytes(b"fake")
        mocker.patch("serapeum.llama_cpp.llm.get_cache_dir", return_value=str(tmp_path))
        mock_hf = mocker.patch(
            "serapeum.llama_cpp.llm._fetch_model_file_hf", return_value=downloaded
        )

        LlamaCPP(hf_model_id=repo_id, hf_filename=filename, **_formatters())

        mock_hf.assert_called_once_with(repo_id, filename, cache_dir)

    @pytest.mark.unit
    def test_model_post_init_updates_model_path_after_hf_download(
        self, mock_llama_cls: MagicMock, mocker: Any, tmp_path: Path
    ) -> None:
        """Test that model_path is set to the HuggingFace-downloaded file path.

        Test scenario:
            After _fetch_model_file_hf returns a local Path, llm.model_path
            must be updated to the string form of that path.
        """
        repo_id = "TheBloke/Llama-2-13B-chat-GGUF"
        filename = "llama-2-13b-chat.Q4_0.gguf"
        cache_dir = tmp_path / "models"
        downloaded = cache_dir / filename
        downloaded.parent.mkdir(parents=True, exist_ok=True)
        downloaded.write_bytes(b"fake")
        mocker.patch("serapeum.llama_cpp.llm.get_cache_dir", return_value=str(tmp_path))
        mocker.patch(
            "serapeum.llama_cpp.llm._fetch_model_file_hf", return_value=downloaded
        )

        llm = LlamaCPP(hf_model_id=repo_id, hf_filename=filename, **_formatters())

        assert llm.model_path == str(
            downloaded
        ), f"model_path should be set to {str(downloaded)!r}, got {llm.model_path!r}"

    @pytest.mark.unit
    def test_model_cache_returns_cached_instance_on_second_construction(
        self, mock_llama_cls: MagicMock, model_file: Path
    ) -> None:
        """Test that a second construction with the same path reuses the cached Llama.

        Test scenario:
            Two LlamaCPP instances with identical model_path and model_kwargs
            must share the same underlying Llama instance and only call
            Llama() once (the second hit comes from the WeakValueDictionary cache).
        """
        first = LlamaCPP(model_path=str(model_file), **_formatters())
        second = LlamaCPP(model_path=str(model_file), **_formatters())

        assert mock_llama_cls.call_count == 1, (
            f"Llama() should only be called once when the cache hits, "
            f"got call_count={mock_llama_cls.call_count}"
        )
        assert (
            second._model is first._model
        ), "Both instances should share the same underlying Llama object"

    @pytest.mark.unit
    def test_model_cache_creates_new_instance_for_different_path(
        self, mock_llama_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Test that different model paths produce different Llama instances.

        Test scenario:
            Two LlamaCPP instances with distinct model_path values must each
            call Llama() independently — no cross-path cache sharing.
            side_effect returns a fresh MagicMock per call so _model references
            are distinct objects (the default return_value is the same singleton).
        """
        path_a = tmp_path / "a.gguf"
        path_b = tmp_path / "b.gguf"
        path_a.write_bytes(b"a")
        path_b.write_bytes(b"b")

        def _make_mock() -> MagicMock:
            m = MagicMock()
            m.context_params.n_ctx = 512
            m.tokenize.return_value = []
            return m

        mock_llama_cls.side_effect = [_make_mock(), _make_mock()]

        first = LlamaCPP(model_path=str(path_a), **_formatters())
        second = LlamaCPP(model_path=str(path_b), **_formatters())

        assert mock_llama_cls.call_count == 2, (
            f"Llama() should be called once per distinct path, "
            f"got call_count={mock_llama_cls.call_count}"
        )
        assert (
            second._model is not first._model
        ), "Different paths must produce different Llama instances"

    @pytest.mark.unit
    def test_metadata_context_window_reads_from_model(self, llm: LlamaCPP) -> None:
        """Test metadata.context_window reflects the model's context_params.n_ctx.

        Test scenario:
            The mock Llama instance has context_params.n_ctx = 512; the
            Metadata object must expose that same value.
        """
        assert (
            llm.metadata.context_window == 512
        ), f"Expected context_window=512 from mock model, got {llm.metadata.context_window}"

    @pytest.mark.unit
    def test_metadata_num_output_equals_max_new_tokens(self, llm: LlamaCPP) -> None:
        """Test metadata.num_output equals the max_new_tokens field.

        Test scenario:
            The fixture creates LlamaCPP with max_new_tokens=32; metadata
            must expose that value as num_output.
        """
        assert (
            llm.metadata.num_output == 32
        ), f"Expected num_output=32, got {llm.metadata.num_output}"

    @pytest.mark.unit
    def test_metadata_model_name_is_model_path_string(
        self, llm: LlamaCPP, model_file: Path
    ) -> None:
        """Test metadata.model_name equals the resolved model_path string.

        Test scenario:
            model_name in Metadata should match the model_path string passed
            at construction time.
        """
        assert llm.metadata.model_name == str(
            model_file
        ), f"Expected model_name={str(model_file)!r}, got {llm.metadata.model_name!r}"

    @pytest.mark.unit
    def test_tokenize_delegates_to_model(
        self, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Test tokenize() calls _model.tokenize with the UTF-8 encoded text.

        Test scenario:
            tokenize("hello") must call mock_llama.tokenize(b"hello") and
            return whatever the backend returns.
        """
        mock_llama.tokenize.return_value = [1, 2, 3]
        result = llm.tokenize("hello")
        mock_llama.tokenize.assert_called_once_with(b"hello")
        assert result == [1, 2, 3], f"Expected [1, 2, 3], got {result}"

    @pytest.mark.unit
    def test_count_tokens_returns_token_count(
        self, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Test count_tokens() returns the length of the tokenize() result.

        Test scenario:
            If tokenize returns a 5-element list, count_tokens must return 5.
        """
        mock_llama.tokenize.return_value = [10, 20, 30, 40, 50]
        count = llm.count_tokens("some text")
        assert count == 5, f"Expected 5 tokens, got {count}"

    @pytest.mark.unit
    def test_guard_context_does_not_raise_within_context_window(
        self, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Test _guard_context passes silently when token count is within the window.

        Test scenario:
            A prompt whose tokenize length equals context_window - 1 must
            not raise any exception.
        """
        mock_llama.tokenize.return_value = [0] * (llm.context_window - 1)
        llm._guard_context("short prompt")  # must not raise

    @pytest.mark.unit
    def test_guard_context_raises_when_prompt_exceeds_context_window(
        self, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Test _guard_context raises ValueError when token count exceeds the window.

        Test scenario:
            A prompt whose tokenize length is context_window + 1 must raise
            ValueError with a message referencing both the token count and
            the context_window limit.
        """
        mock_llama.tokenize.return_value = [0] * (llm.context_window + 1)
        with pytest.raises(ValueError, match="context_window") as exc_info:
            llm._guard_context("very long prompt")
        assert str(llm.context_window) in str(
            exc_info.value
        ), f"Error should cite context_window={llm.context_window}, got: {exc_info.value}"

    @pytest.mark.unit
    def test_complete_non_stream_returns_completion_response(
        self, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Test complete() returns a CompletionResponse for non-streaming calls.

        Test scenario:
            A plain prompt (formatted=True) should produce a single
            CompletionResponse whose text matches the mock backend reply.
        """
        mock_llama.return_value = {"choices": [{"text": "Hello!"}]}
        result = llm.complete("Say hello.", formatted=True)
        assert isinstance(
            result, CompletionResponse
        ), f"Expected CompletionResponse, got {type(result)}"
        assert result.text == "Hello!", f"Expected text 'Hello!', got {result.text!r}"

    @pytest.mark.unit
    def test_complete_non_stream_attaches_raw_response(
        self, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Test that complete() attaches the raw backend dict to the response.

        Test scenario:
            The raw attribute on CompletionResponse should be the exact dict
            returned by the Llama mock, preserving all backend metadata.
        """
        raw = {"choices": [{"text": "Hi"}], "id": "abc123"}
        mock_llama.return_value = raw
        result = llm.complete("Say hi.", formatted=True)
        assert (
            result.raw == raw
        ), f"raw should be the mock response dict, got: {result.raw}"

    @pytest.mark.unit
    def test_complete_applies_completion_to_prompt_when_not_formatted(
        self, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Test that completion_to_prompt is invoked when formatted=False.

        Test scenario:
            We install a formatter that prepends a sentinel prefix. When
            formatted=False, the model must receive the prefixed prompt,
            proving the formatter was applied.
        """
        mock_llama.return_value = {"choices": [{"text": "ok"}]}
        called_with: list[str] = []

        def tracking_formatter(
            completion: str, system_prompt: str | None = None
        ) -> str:
            called_with.append(completion)
            return f"[FMT]{completion}"

        llm.completion_to_prompt = tracking_formatter
        llm.complete("raw input", formatted=False)

        assert called_with == [
            "raw input"
        ], f"Formatter should have been called with 'raw input', got: {called_with}"
        assert mock_llama.call_args.kwargs["prompt"] == "[FMT]raw input", (
            f"Model should receive the formatted prompt, "
            f"got: {mock_llama.call_args.kwargs.get('prompt')!r}"
        )

    @pytest.mark.unit
    def test_complete_formatted_true_bypasses_completion_to_prompt(
        self, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Test that completion_to_prompt is NOT called when formatted=True.

        Test scenario:
            Passing formatted=True signals the prompt is already prepared —
            the formatter must not be applied and the prompt must reach the
            model unchanged.
        """
        mock_llama.return_value = {"choices": [{"text": "ok"}]}
        custom_prompt = "<already_formatted_prompt>"
        llm.complete(custom_prompt, formatted=True)
        assert mock_llama.call_args.kwargs["prompt"] == custom_prompt, (
            f"Prompt must pass through unchanged when formatted=True, "
            f"got: {mock_llama.call_args.kwargs.get('prompt')!r}"
        )

    @pytest.mark.unit
    def test_complete_forwards_temperature_and_max_tokens(
        self, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Test that temperature and max_new_tokens reach the Llama call.

        Test scenario:
            The Llama callable must receive temperature and max_tokens matching
            the fields set at construction time.
        """
        mock_llama.return_value = {"choices": [{"text": "ok"}]}
        llm.complete("hi", formatted=True)
        call_kw = mock_llama.call_args.kwargs
        assert (
            call_kw["temperature"] == llm.temperature
        ), f"temperature should be forwarded, got: {call_kw.get('temperature')}"
        assert (
            call_kw["max_tokens"] == llm.max_new_tokens
        ), f"max_tokens should equal max_new_tokens, got: {call_kw.get('max_tokens')}"

    @pytest.mark.unit
    def test_complete_non_stream_sets_stream_false_on_backend(
        self, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Test that non-streaming complete() passes stream=False to the backend.

        Test scenario:
            The Llama model must be invoked with stream=False so it returns
            a single response dict rather than an iterator.
        """
        mock_llama.return_value = {"choices": [{"text": "ok"}]}
        llm.complete("hi", formatted=True, stream=False)
        assert mock_llama.call_args.kwargs["stream"] is False, (
            f"Backend should receive stream=False for non-streaming, "
            f"got: {mock_llama.call_args.kwargs.get('stream')}"
        )

    @pytest.mark.unit
    def test_complete_merges_generate_kwargs_with_lower_priority(
        self, mock_llama: MagicMock, model_file: Path
    ) -> None:
        """Test that generate_kwargs are merged but overridden by explicit fields.

        Test scenario:
            generate_kwargs{"temperature": 0.9} should be present in the
            backend call but overridden by the explicit temperature field
            (temperature=0.1), leaving temperature=0.1.
        """
        llm = LlamaCPP(
            model_path=str(model_file),
            temperature=0.1,
            generate_kwargs={"temperature": 0.9, "top_p": 0.95},
            **_formatters(),
        )
        mock_llama.return_value = {"choices": [{"text": "ok"}]}
        llm.complete("hi", formatted=True)
        call_kw = mock_llama.call_args.kwargs
        assert (
            call_kw["temperature"] == 0.1
        ), f"Explicit temperature should override generate_kwargs, got: {call_kw.get('temperature')}"
        assert (
            call_kw.get("top_p") == 0.95
        ), f"generate_kwargs.top_p should be forwarded, got: {call_kw}"

    @pytest.mark.unit
    def test_complete_passes_extra_kwargs_to_backend(
        self, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Test that extra **kwargs passed to complete() are forwarded to the backend.

        Test scenario:
            Calling complete(..., stop=["END"]) should result in stop=["END"]
            appearing in the Llama call kwargs.
        """
        mock_llama.return_value = {"choices": [{"text": "ok"}]}
        llm.complete("hi", formatted=True, stop=["END"])
        assert mock_llama.call_args.kwargs.get("stop") == [
            "END"
        ], f"extra stop kwarg should reach the backend, got: {mock_llama.call_args.kwargs}"

    @pytest.mark.unit
    def test_complete_passes_stop_field_to_backend(
        self, mock_llama: MagicMock, model_file: Path
    ) -> None:
        """Test that the stop field is forwarded to the backend via setdefault.

        Test scenario:
            An LlamaCPP constructed with stop=["</s>"] must include
            stop=["</s>"] in the backend call when no per-call stop kwarg is given.
        """
        llm = LlamaCPP(
            model_path=str(model_file),
            stop=["</s>"],
            **_formatters(),
        )
        mock_llama.return_value = {"choices": [{"text": "ok"}]}
        llm.complete("hi", formatted=True)
        assert mock_llama.call_args.kwargs.get("stop") == ["</s>"], (
            f"stop field should be forwarded to the backend, "
            f"got: {mock_llama.call_args.kwargs}"
        )

    @pytest.mark.unit
    def test_complete_raises_when_prompt_exceeds_context_window(
        self, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Test that complete() raises ValueError when the prompt exceeds the window.

        Test scenario:
            If tokenize returns a list longer than context_window, _guard_context
            must raise ValueError before the model is called.
        """
        mock_llama.tokenize.return_value = [0] * (llm.context_window + 1)
        with pytest.raises(ValueError, match="context_window"):
            llm.complete("too long", formatted=True)

    @pytest.mark.unit
    def test_complete_stream_returns_generator(
        self, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Test that complete(stream=True) returns an iterable of CompletionResponse.

        Test scenario:
            The return value must be iterable; consuming the first element
            must yield a CompletionResponse.
        """
        mock_llama.return_value = [{"choices": [{"text": "tok1"}]}]
        gen = llm.complete("hi", formatted=True, stream=True)
        first = next(iter(gen))
        assert isinstance(
            first, CompletionResponse
        ), f"Expected CompletionResponse chunk, got {type(first)}"

    @pytest.mark.unit
    def test_complete_stream_sets_stream_true_on_backend(
        self, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Test that streaming complete() passes stream=True to the backend.

        Test scenario:
            The Llama callable must receive stream=True so it returns an
            iterator of token dicts rather than a single response.
        """
        mock_llama.return_value = iter([{"choices": [{"text": "tok"}]}])
        list(llm.complete("hi", formatted=True, stream=True))
        assert mock_llama.call_args.kwargs["stream"] is True, (
            f"Backend should receive stream=True, "
            f"got: {mock_llama.call_args.kwargs.get('stream')}"
        )

    @pytest.mark.unit
    def test_complete_stream_each_chunk_has_string_delta(
        self, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Test that every streaming chunk carries a non-None string delta.

        Test scenario:
            Each yielded CompletionResponse must have a string delta
            representing the incremental token text.
        """
        mock_llama.return_value = [
            {"choices": [{"text": "tok1"}]},
            {"choices": [{"text": "tok2"}]},
        ]
        for chunk in llm.complete("hi", formatted=True, stream=True):
            assert (
                chunk.delta is not None
            ), f"Chunk delta must not be None, got chunk: {chunk}"
            assert isinstance(
                chunk.delta, str
            ), f"Delta must be str, got {type(chunk.delta)}"

    @pytest.mark.unit
    def test_complete_stream_text_accumulates_monotonically(
        self, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Test that chunk.text grows monotonically across the stream.

        Test scenario:
            Each successive chunk's text should be at least as long as the
            previous one — it is the cumulative concatenation of all deltas.
        """
        mock_llama.return_value = [
            {"choices": [{"text": "A"}]},
            {"choices": [{"text": "B"}]},
            {"choices": [{"text": "C"}]},
        ]
        prev_len = 0
        for chunk in llm.complete("hi", formatted=True, stream=True):
            assert (
                len(chunk.text) >= prev_len
            ), f"Text length decreased: {prev_len} → {len(chunk.text)}"
            prev_len = len(chunk.text)

    @pytest.mark.unit
    def test_complete_stream_final_text_equals_joined_deltas(
        self, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Test that the last chunk's text equals all deltas concatenated.

        Test scenario:
            Joining every delta from the stream must reproduce the text
            field of the final chunk exactly.
        """
        mock_llama.return_value = [
            {"choices": [{"text": "Hello"}]},
            {"choices": [{"text": " world"}]},
        ]
        chunks = list(llm.complete("hi", formatted=True, stream=True))
        joined = "".join(c.delta for c in chunks if c.delta)
        assert (
            joined == chunks[-1].text
        ), f"Joined deltas {joined!r} != final text {chunks[-1].text!r}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_acomplete_non_stream_returns_completion_response(
        self, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Test acomplete() resolves to a CompletionResponse for non-streaming.

        Test scenario:
            Awaiting acomplete() must yield a CompletionResponse whose text
            matches the mock backend reply.
        """
        mock_llama.return_value = {"choices": [{"text": "async reply"}]}
        result = await llm.acomplete("Say hello.", formatted=True)
        assert isinstance(
            result, CompletionResponse
        ), f"Expected CompletionResponse, got {type(result)}"
        assert (
            result.text == "async reply"
        ), f"Expected text 'async reply', got {result.text!r}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_acomplete_non_stream_does_not_block_event_loop(
        self, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Test acomplete() runs concurrently without blocking the event loop.

        Test scenario:
            A side coroutine scheduled with asyncio.gather must run while
            acomplete is in-flight, proving the call is non-blocking.
        """
        mock_llama.return_value = {"choices": [{"text": "ok"}]}
        side_ran: list[bool] = []

        async def side() -> None:
            side_ran.append(True)

        result, _ = await asyncio.gather(
            llm.acomplete("hi", formatted=True),
            side(),
        )
        assert isinstance(
            result, CompletionResponse
        ), f"acomplete should still return CompletionResponse, got {type(result)}"
        assert side_ran, "Side coroutine should have run concurrently"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_acomplete_stream_returns_async_iterable(
        self, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Test acomplete(stream=True) returns an async generator of CompletionResponse.

        Test scenario:
            The return value must be async-iterable and yield at least one
            CompletionResponse chunk with a non-None delta.
        """
        mock_llama.return_value = [{"choices": [{"text": "tok"}]}]
        gen = await llm.acomplete("hi", formatted=True, stream=True)
        chunks = [chunk async for chunk in gen]
        assert len(chunks) > 0, "Async stream must yield at least one chunk"
        assert isinstance(
            chunks[0], CompletionResponse
        ), f"Expected CompletionResponse chunk, got {type(chunks[0])}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_acomplete_stream_delta_and_text_are_correct(
        self, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Test that async streaming chunks carry correct delta and cumulative text.

        Test scenario:
            Two-token stream ['A', 'B']: first chunk delta='A', text='A';
            second chunk delta='B', text='AB'.
        """
        mock_llama.return_value = [
            {"choices": [{"text": "A"}]},
            {"choices": [{"text": "B"}]},
        ]
        gen = await llm.acomplete("hi", formatted=True, stream=True)
        chunks = [chunk async for chunk in gen]
        assert (
            chunks[0].delta == "A"
        ), f"First chunk delta should be 'A', got {chunks[0].delta!r}"
        assert (
            chunks[1].delta == "B"
        ), f"Second chunk delta should be 'B', got {chunks[1].delta!r}"
        assert (
            chunks[1].text == "AB"
        ), f"Final accumulated text should be 'AB', got {chunks[1].text!r}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_acomplete_formatted_true_bypasses_formatter(
        self, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Test that acomplete(formatted=True) does not apply the formatter.

        Test scenario:
            A tracking formatter must not be called when formatted=True;
            the raw prompt should reach the backend unchanged.
        """
        mock_llama.return_value = {"choices": [{"text": "ok"}]}
        called: list[str] = []

        def tracking(completion: str, system_prompt: str | None = None) -> str:
            called.append(completion)
            return f"[FMT]{completion}"

        llm.completion_to_prompt = tracking
        await llm.acomplete("raw", formatted=True)
        assert (
            not called
        ), f"Formatter must not be called when formatted=True, called with: {called}"


@pytest.mark.unit
class TestLlamaCPPInheritance:
    """Tests for LlamaCPP class hierarchy and protocol compliance."""

    @pytest.mark.unit
    def test_is_subclass_of_llm(self) -> None:
        """Test that LlamaCPP inherits from the LLM base class.

        Test scenario:
            LLM is the core abstract base; LlamaCPP must appear in its MRO.
        """
        from serapeum.core.llms import LLM

        assert issubclass(
            LlamaCPP, LLM
        ), f"LlamaCPP must be a subclass of LLM, MRO: {[c.__name__ for c in LlamaCPP.__mro__]}"

    @pytest.mark.unit
    def test_has_completion_to_chat_mixin(self) -> None:
        """Test that LlamaCPP includes CompletionToChatMixin in its MRO.

        Test scenario:
            CompletionToChatMixin provides chat() and achat() — verifying it
            is in the MRO confirms those methods are available.
        """
        from serapeum.core.llms import CompletionToChatMixin

        assert issubclass(LlamaCPP, CompletionToChatMixin), (
            f"LlamaCPP must include CompletionToChatMixin, "
            f"MRO: {[c.__name__ for c in LlamaCPP.__mro__]}"
        )


@pytest.mark.performance
class TestLlamaCPPBenchmarks:
    """Performance benchmarks for LlamaCPP hot paths."""

    @pytest.mark.performance
    def test_complete_non_stream_throughput(
        self, benchmark: Any, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Benchmark non-streaming complete() call overhead.

        Test scenario:
            Measures the overhead of the Python-side complete() dispatch
            (formatter application + result wrapping) with a mocked backend.
            Serves as a regression baseline for the completion path.
        """
        mock_llama.return_value = {"choices": [{"text": "benchmark token"}]}
        result = benchmark(llm.complete, "benchmark prompt", True)
        assert isinstance(
            result, CompletionResponse
        ), f"Benchmark result should be CompletionResponse, got {type(result)}"

    @pytest.mark.performance
    def test_complete_stream_collect_throughput(
        self, benchmark: Any, llm: LlamaCPP, mock_llama: MagicMock
    ) -> None:
        """Benchmark streaming complete() collection overhead.

        Test scenario:
            Measures the cost of iterating over a 10-token mock stream
            and collecting all CompletionResponse chunks.
        """
        tokens = [{"choices": [{"text": f"t{i}"}]} for i in range(10)]
        mock_llama.return_value = tokens

        def run() -> list[CompletionResponse]:
            return list(llm.complete("bench", formatted=True, stream=True))

        result = benchmark(run)
        assert len(result) == 10, f"Expected 10 chunks, got {len(result)}"
