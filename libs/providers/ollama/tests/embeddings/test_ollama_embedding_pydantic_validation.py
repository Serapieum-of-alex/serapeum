
"""Unit tests for OllamaEmbedding Pydantic validation and instantiation."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from pydantic import ValidationError

from serapeum.ollama.embedding import OllamaEmbedding


class TestPydanticInstantiation:
    """Test suite for Pydantic-style instantiation patterns."""

    def test_dict_fields_reject_none_values(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            OllamaEmbedding(model_name="test", client_kwargs=None)

        assert "client_kwargs" in str(exc_info.value)
        assert "dict" in str(exc_info.value).lower()

    def test_dict_fields_reject_none_for_ollama_additional_kwargs(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            OllamaEmbedding(model_name="test", ollama_additional_kwargs=None)

        assert "ollama_additional_kwargs" in str(exc_info.value)

    def test_multiple_dict_fields_reject_none_simultaneously(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            OllamaEmbedding(
                model_name="test",
                client_kwargs=None,
                ollama_additional_kwargs=None,
            )

        errors = str(exc_info.value)
        assert "client_kwargs" in errors
        assert "ollama_additional_kwargs" in errors

    def test_dict_fields_accept_empty_dicts(self) -> None:
        embedder = OllamaEmbedding(
            model_name="test",
            client_kwargs={},
            ollama_additional_kwargs={},
        )

        assert embedder.client_kwargs == {}
        assert embedder.ollama_additional_kwargs == {}

    def test_dict_fields_use_default_factory_when_not_provided(self) -> None:
        embedder = OllamaEmbedding(model_name="test")

        assert embedder.client_kwargs == {}
        assert embedder.ollama_additional_kwargs == {}
        assert isinstance(embedder.client_kwargs, dict)
        assert isinstance(embedder.ollama_additional_kwargs, dict)

    def test_dict_fields_accept_populated_dicts(self) -> None:
        client_kwargs = {"timeout": 30, "headers": {"Auth": "token"}}
        ollama_kwargs = {"temperature": 0.7, "top_p": 0.9}

        embedder = OllamaEmbedding(
            model_name="test",
            client_kwargs=client_kwargs,
            ollama_additional_kwargs=ollama_kwargs,
        )

        assert embedder.client_kwargs == client_kwargs
        assert embedder.ollama_additional_kwargs == ollama_kwargs

    def test_base_url_uses_default_when_not_provided(self) -> None:
        embedder = OllamaEmbedding(model_name="test")

        assert embedder.base_url == "http://localhost:11434"

    def test_base_url_accepts_custom_value(self) -> None:
        custom_url = "http://custom-host:8080"
        embedder = OllamaEmbedding(model_name="test", base_url=custom_url)

        assert embedder.base_url == custom_url

    def test_model_name_is_required_field(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            OllamaEmbedding()

        assert "model_name" in str(exc_info.value)


class TestModelValidatorBehavior:
    """Test suite for client initialization logic (lazy via properties)."""

    def test_model_validator_initializes_sync_client(self) -> None:
        with (
            patch("serapeum.ollama.client.ollama_sdk.Client") as mock_client_cls,
            patch("serapeum.ollama.client.ollama_sdk.AsyncClient"),
        ):
            embedder = OllamaEmbedding(model_name="test")
            _ = embedder.client  # trigger lazy init

            assert hasattr(embedder, "_client")
            assert embedder._client is not None
            mock_client_cls.assert_called_once_with(host="http://localhost:11434")

    def test_model_validator_initializes_async_client(self) -> None:
        with (
            patch("serapeum.ollama.client.ollama_sdk.Client"),
            patch("serapeum.ollama.client.ollama_sdk.AsyncClient") as mock_async_cls,
        ):
            embedder = OllamaEmbedding(model_name="test")
            _ = embedder.async_client  # trigger lazy init

            assert hasattr(embedder, "_async_client")
            assert embedder._async_client is not None
            mock_async_cls.assert_called_once_with(host="http://localhost:11434")

    def test_model_validator_passes_base_url_to_clients(self) -> None:
        with (
            patch("serapeum.ollama.client.ollama_sdk.Client") as mock_client_cls,
            patch("serapeum.ollama.client.ollama_sdk.AsyncClient") as mock_async_cls,
        ):
            custom_url = "http://custom:9999"
            embedder = OllamaEmbedding(model_name="test", base_url=custom_url)
            _ = embedder.client
            _ = embedder.async_client

            mock_client_cls.assert_called_once_with(host=custom_url)
            mock_async_cls.assert_called_once_with(host=custom_url)

    def test_model_validator_passes_client_kwargs_to_clients(self) -> None:
        with (
            patch("serapeum.ollama.client.ollama_sdk.Client") as mock_client_cls,
            patch("serapeum.ollama.client.ollama_sdk.AsyncClient") as mock_async_cls,
        ):
            kwargs = {"timeout": 60, "headers": {"X-Custom": "value"}}
            embedder = OllamaEmbedding(model_name="test", client_kwargs=kwargs)
            _ = embedder.client
            _ = embedder.async_client

            mock_client_cls.assert_called_once_with(
                host="http://localhost:11434", timeout=60, headers={"X-Custom": "value"}
            )
            mock_async_cls.assert_called_once_with(
                host="http://localhost:11434", timeout=60, headers={"X-Custom": "value"}
            )

    def test_api_key_auth_header_preserved_when_client_kwargs_also_has_headers(self) -> None:
        with (
            patch("serapeum.ollama.client.ollama_sdk.Client") as mock_client_cls,
            patch("serapeum.ollama.client.ollama_sdk.AsyncClient") as mock_async_cls,
        ):
            embedder = OllamaEmbedding(
                model_name="test",
                api_key="secret",
                client_kwargs={"headers": {"X-Custom": "value"}},
            )
            _ = embedder.client
            _ = embedder.async_client

            expected_headers = {"Authorization": "Bearer secret", "X-Custom": "value"}
            mock_client_cls.assert_called_once_with(
                host="https://api.ollama.com", headers=expected_headers
            )
            mock_async_cls.assert_called_once_with(
                host="https://api.ollama.com", headers=expected_headers
            )

    def test_client_kwargs_headers_take_precedence_over_base_headers(self) -> None:
        with (
            patch("serapeum.ollama.client.ollama_sdk.Client") as mock_client_cls,
            patch("serapeum.ollama.client.ollama_sdk.AsyncClient") as mock_async_cls,
        ):
            embedder = OllamaEmbedding(
                model_name="test",
                api_key="secret",
                client_kwargs={"headers": {"Authorization": "Bearer override"}},
            )
            _ = embedder.client
            _ = embedder.async_client

            expected_headers = {"Authorization": "Bearer override"}
            mock_client_cls.assert_called_once_with(
                host="https://api.ollama.com", headers=expected_headers
            )
            mock_async_cls.assert_called_once_with(
                host="https://api.ollama.com", headers=expected_headers
            )

    def test_model_validator_handles_empty_client_kwargs(self) -> None:
        with (
            patch("serapeum.ollama.client.ollama_sdk.Client") as mock_client_cls,
            patch("serapeum.ollama.client.ollama_sdk.AsyncClient") as mock_async_cls,
        ):
            embedder = OllamaEmbedding(model_name="test", client_kwargs={})
            _ = embedder.client
            _ = embedder.async_client

            mock_client_cls.assert_called_once_with(host="http://localhost:11434")
            mock_async_cls.assert_called_once_with(host="http://localhost:11434")

    def test_model_validator_runs_after_field_validation(self) -> None:
        with (
            patch("serapeum.ollama.client.ollama_sdk.Client"),
            patch("serapeum.ollama.client.ollama_sdk.AsyncClient"),
        ):
            embedder = OllamaEmbedding(
                model_name="test",
                base_url="https://host:1234",
                batch_size=50,
            )
            _ = embedder.client
            _ = embedder.async_client

            assert embedder.model_name == "test"
            assert embedder.base_url == "https://host:1234"
            assert embedder.batch_size == 50
            assert embedder._client is not None
            assert embedder._async_client is not None

    def test_model_validator_preserves_all_field_values(self) -> None:
        embedder = OllamaEmbedding(
            model_name="test-model",
            base_url="http://custom:8080",
            batch_size=100,
            ollama_additional_kwargs={"temp": 0.5},
            query_instruction="Query: ",
            text_instruction="Text: ",
            keep_alive="10m",
            client_kwargs={"timeout": 30},
        )

        assert embedder.model_name == "test-model"
        assert embedder.base_url == "http://custom:8080"
        assert embedder.batch_size == 100
        assert embedder.ollama_additional_kwargs == {"temp": 0.5}
        assert embedder.query_instruction == "Query: "
        assert embedder.text_instruction == "Text: "
        assert embedder.keep_alive == "10m"
        assert embedder.client_kwargs == {"timeout": 30}


class TestPydanticSerialization:
    """Test suite for Pydantic serialization methods."""

    def test_model_dump_returns_dict_with_all_fields(self) -> None:
        embedder = OllamaEmbedding(
            model_name="test",
            base_url="http://localhost:11434",
            batch_size=50,
        )

        data = embedder.model_dump()

        assert isinstance(data, dict)
        assert data["model_name"] == "test"
        assert data["base_url"] == "http://localhost:11434"
        assert data["batch_size"] == 50

    def test_model_dump_includes_optional_fields_when_set(self) -> None:
        embedder = OllamaEmbedding(
            model_name="test",
            query_instruction="Query: ",
            text_instruction="Text: ",
        )

        data = embedder.model_dump()

        assert data["query_instruction"] == "Query: "
        assert data["text_instruction"] == "Text: "

    def test_model_dump_excludes_private_attributes(self) -> None:
        embedder = OllamaEmbedding(model_name="test")

        data = embedder.model_dump()

        assert "_client" not in data
        assert "_async_client" not in data

    def test_model_validate_recreates_instance_from_dict(self) -> None:
        original = OllamaEmbedding(
            model_name="test",
            base_url="http://custom:8080",
            batch_size=75,
        )

        data = original.model_dump()

        recreated = OllamaEmbedding.model_validate(data)

        assert recreated.model_name == original.model_name
        assert recreated.base_url == original.base_url
        assert recreated.batch_size == original.batch_size

    def test_model_validate_reinitialize_private_attributes(self) -> None:
        with (
            patch("serapeum.ollama.client.ollama_sdk.Client") as mock_client_cls,
            patch("serapeum.ollama.client.ollama_sdk.AsyncClient") as mock_async_cls,
        ):
            original = OllamaEmbedding(model_name="test")
            data = original.model_dump()

            mock_client_cls.reset_mock()
            mock_async_cls.reset_mock()

            recreated = OllamaEmbedding.model_validate(data)
            _ = recreated.client
            _ = recreated.async_client

            assert hasattr(recreated, "_client")
            assert hasattr(recreated, "_async_client")
            assert recreated._client is not None
            assert recreated._async_client is not None

    def test_model_dump_json_produces_json_string(self) -> None:
        embedder = OllamaEmbedding(model_name="test", batch_size=100)

        json_str = embedder.model_dump_json()

        assert isinstance(json_str, str)
        assert '"model_name"' in json_str or "'model_name'" in json_str
        assert "test" in json_str

    def test_model_validate_json_recreates_from_json_string(self) -> None:
        original = OllamaEmbedding(
            model_name="test",
            base_url="https://host:1234",
        )

        json_str = original.model_dump_json()
        recreated = OllamaEmbedding.model_validate_json(json_str)

        assert recreated.model_name == original.model_name
        assert recreated.base_url == original.base_url

    def test_serialization_roundtrip_preserves_all_data(self) -> None:
        original = OllamaEmbedding(
            model_name="llama2",
            base_url="http://custom:9090",
            batch_size=200,
            ollama_additional_kwargs={"temperature": 0.8},
            query_instruction="Search: ",
            text_instruction="Document: ",
            keep_alive="15m",
        )

        data = original.model_dump()
        recreated = OllamaEmbedding.model_validate(data)

        assert recreated.model_name == original.model_name
        assert recreated.base_url == original.base_url
        assert recreated.batch_size == original.batch_size
        assert (
            recreated.ollama_additional_kwargs == original.ollama_additional_kwargs
        )
        assert recreated.query_instruction == original.query_instruction
        assert recreated.text_instruction == original.text_instruction
        assert recreated.keep_alive == original.keep_alive


class TestFieldValidation:
    """Test suite for Pydantic field-level validation."""

    def test_batch_size_rejects_zero(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            OllamaEmbedding(model_name="test", batch_size=0)

        assert "batch_size" in str(exc_info.value)

    def test_batch_size_rejects_negative_values(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            OllamaEmbedding(model_name="test", batch_size=-10)

        assert "batch_size" in str(exc_info.value)

    def test_batch_size_rejects_values_over_2048(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            OllamaEmbedding(model_name="test", batch_size=2049)

        assert "batch_size" in str(exc_info.value)

    def test_batch_size_accepts_boundary_value_1(self) -> None:
        embedder = OllamaEmbedding(model_name="test", batch_size=1)

        assert embedder.batch_size == 1

    def test_batch_size_accepts_boundary_value_2048(self) -> None:
        embedder = OllamaEmbedding(model_name="test", batch_size=2048)

        assert embedder.batch_size == 2048

    def test_batch_size_accepts_valid_middle_values(self) -> None:
        embedder = OllamaEmbedding(model_name="test", batch_size=512)

        assert embedder.batch_size == 512

    def test_keep_alive_accepts_string_values(self) -> None:
        embedder = OllamaEmbedding(model_name="test", keep_alive="10m")

        assert embedder.keep_alive == "10m"

    def test_keep_alive_accepts_float_values(self) -> None:
        embedder = OllamaEmbedding(model_name="test", keep_alive=5.5)

        assert embedder.keep_alive == 5.5

    def test_keep_alive_accepts_none_value(self) -> None:
        embedder = OllamaEmbedding(model_name="test", keep_alive=None)

        assert embedder.keep_alive is None

    def test_query_instruction_accepts_none_as_default(self) -> None:
        embedder = OllamaEmbedding(model_name="test")

        assert embedder.query_instruction is None

    def test_text_instruction_accepts_none_as_default(self) -> None:
        embedder = OllamaEmbedding(model_name="test")

        assert embedder.text_instruction is None

    def test_instructions_accept_string_values(self) -> None:
        embedder = OllamaEmbedding(
            model_name="test",
            query_instruction="Query instruction",
            text_instruction="Text instruction",
        )

        assert embedder.query_instruction == "Query instruction"
        assert embedder.text_instruction == "Text instruction"

    def test_instructions_accept_empty_strings(self) -> None:
        embedder = OllamaEmbedding(
            model_name="test",
            query_instruction="",
            text_instruction="",
        )

        assert embedder.query_instruction == ""
        assert embedder.text_instruction == ""


class TestClientKwargsIsolation:
    """Test suite for client_kwargs isolation between instances."""

    def test_separate_instances_have_independent_client_kwargs(self) -> None:
        embedder1 = OllamaEmbedding(model_name="test1")
        embedder2 = OllamaEmbedding(model_name="test2")

        embedder1.client_kwargs["custom"] = "value1"

        assert "custom" not in embedder2.client_kwargs
        assert embedder1.client_kwargs != embedder2.client_kwargs

    def test_separate_instances_have_independent_ollama_kwargs(self) -> None:
        embedder1 = OllamaEmbedding(model_name="test1")
        embedder2 = OllamaEmbedding(model_name="test2")

        embedder1.ollama_additional_kwargs["temperature"] = 0.8

        assert "temperature" not in embedder2.ollama_additional_kwargs
        assert (
            embedder1.ollama_additional_kwargs != embedder2.ollama_additional_kwargs
        )

    def test_modifying_client_kwargs_after_init_does_not_affect_clients(self) -> None:
        with (
            patch("serapeum.ollama.client.ollama_sdk.Client"),
            patch("serapeum.ollama.client.ollama_sdk.AsyncClient"),
        ):
            embedder = OllamaEmbedding(model_name="test", client_kwargs={"timeout": 30})
            original_client = embedder.client  # trigger lazy init
            embedder.client_kwargs["new_key"] = "new_value"

            assert embedder.client is original_client

    def test_default_factory_creates_new_dict_per_instance(self) -> None:
        embedder1 = OllamaEmbedding(model_name="test1")
        embedder2 = OllamaEmbedding(model_name="test2")
        embedder3 = OllamaEmbedding(model_name="test3")

        id1 = id(embedder1.client_kwargs)
        id2 = id(embedder2.client_kwargs)
        id3 = id(embedder3.client_kwargs)

        assert id1 != id2
        assert id2 != id3
        assert id1 != id3
