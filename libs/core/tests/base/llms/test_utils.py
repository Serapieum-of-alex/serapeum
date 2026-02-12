"""Tests for serapeum.core.base.llms.utils utility functions."""

import pytest

from serapeum.core.base.llms.utils import (
    get_from_param_or_env,
)


class TestGetFromParamOrEnv:
    """Test get_from_param_or_env utility function."""

    def test_param_takes_precedence_over_env_and_default(self, monkeypatch):
        """Test param takes precedence over env and default.

        Inputs: param="VAL", env_key set in environment to another value, default="DEF".
        Expected: Function returns param value regardless of env/default.
        Checks: Exact equality "VAL".
        """
        monkeypatch.setenv("MY_KEY", "ENV_VAL")
        assert (
            get_from_param_or_env("k", param="VAL", env_key="MY_KEY", default="DEF")
            == "VAL"
        )

    def test_env_used_when_param_none_and_env_present(self, monkeypatch):
        """Test environment value when param is None.

        Inputs: param=None, env_key present with non-empty value, default set.
        Expected: Returns environment value.
        Checks: Exact string from environ.
        """
        monkeypatch.setenv("API_TOKEN", "token123")
        assert (
            get_from_param_or_env(
                "token", param=None, env_key="API_TOKEN", default="zzz"
            )
            == "token123"
        )

    def test_default_used_when_param_none_env_missing_or_empty(self, monkeypatch):
        """Test default value when all inputs are missing.

        Inputs: param=None, env_key missing or empty, default provided.
        Expected: Returns default string.
        Checks: Exact match to provided default.
        """
        # ensure env var not present or empty
        monkeypatch.delenv("EMPTY_KEY", raising=False)
        assert (
            get_from_param_or_env("x", param=None, env_key="EMPTY_KEY", default="D")
            == "D"
        )
        monkeypatch.setenv("EMPTY_KEY", "")
        assert (
            get_from_param_or_env("x", param=None, env_key="EMPTY_KEY", default="D2")
            == "D2"
        )

    def test_raises_when_all_missing_with_message(self, monkeypatch):
        """Test error message when all inputs are missing.

        Inputs: No param, no env value, no default.
        Expected: Raises ValueError with a message guiding to set env or pass param containing the key name and env key placeholder.
        Checks: Use regex to match both the key and env variable name in the error message.
        """
        monkeypatch.delenv("NOT_SET", raising=False)
        with pytest.raises(
            ValueError, match=r"Did not find secret,.*`NOT_SET`.*`secret`"
        ):
            get_from_param_or_env("secret", param=None, env_key="NOT_SET", default=None)
