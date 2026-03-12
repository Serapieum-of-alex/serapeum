from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from serapeum.azure_openai.utils import (
    refresh_openai_azure_ad_token,
    resolve_from_aliases,
)


@pytest.mark.unit
class TestResolveFromAliases:
    """Tests for resolve_from_aliases — returns the first non-None argument."""

    def test_all_none_returns_none(self) -> None:
        """All None arguments returns None."""
        assert resolve_from_aliases(None, None, None) is None

    def test_first_non_none_returned(self) -> None:
        """First non-None value is returned, rest are ignored."""
        assert resolve_from_aliases("first", "second", None) == "first"

    def test_middle_non_none_returned(self) -> None:
        """When first arg is None, returns the next non-None value."""
        assert resolve_from_aliases(None, "middle", "last") == "middle"

    def test_last_non_none_returned(self) -> None:
        """When only last arg is non-None, returns it."""
        assert resolve_from_aliases(None, None, "last") == "last"

    def test_single_arg(self) -> None:
        """Single non-None argument is returned."""
        assert resolve_from_aliases("only") == "only"

    def test_no_args_returns_none(self) -> None:
        """No arguments returns None."""
        assert resolve_from_aliases() is None


@pytest.mark.unit
class TestRefreshOpenaiAzureadToken:
    """Tests for refresh_openai_azuread_token — Azure AD token management."""

    def test_valid_non_expired_token_returned_as_is(self) -> None:
        """A token that is not about to expire is returned unchanged."""
        token = MagicMock()
        token.expires_on = time.time() + 300  # expires in 5 minutes
        result = refresh_openai_azure_ad_token(token)
        assert result is token

    @patch("serapeum.azure_openai.utils.DefaultAzureCredential")
    def test_none_token_triggers_refresh(self, mock_credential_cls: MagicMock) -> None:
        """None token triggers credential refresh."""
        mock_credential = mock_credential_cls.return_value
        new_token = MagicMock()
        new_token.expires_on = time.time() + 3600
        mock_credential.get_token.return_value = new_token

        result = refresh_openai_azure_ad_token(None)
        assert result is new_token
        mock_credential.get_token.assert_called_once_with(
            "https://cognitiveservices.azure.com/.default"
        )

    @patch("serapeum.azure_openai.utils.DefaultAzureCredential")
    def test_expired_token_triggers_refresh(
        self, mock_credential_cls: MagicMock
    ) -> None:
        """A token expiring within 60 seconds triggers refresh."""
        old_token = MagicMock()
        old_token.expires_on = time.time() + 30  # expires in 30s (< 60s threshold)
        mock_credential = mock_credential_cls.return_value
        new_token = MagicMock()
        new_token.expires_on = time.time() + 3600
        mock_credential.get_token.return_value = new_token

        result = refresh_openai_azure_ad_token(old_token)
        assert result is new_token

    @patch("serapeum.azure_openai.utils.DefaultAzureCredential")
    def test_auth_error_raises_value_error(
        self, mock_credential_cls: MagicMock
    ) -> None:
        """ClientAuthenticationError is wrapped in a ValueError."""
        from azure.core.exceptions import ClientAuthenticationError

        mock_credential = mock_credential_cls.return_value
        mock_credential.get_token.side_effect = ClientAuthenticationError(
            message="bad creds"
        )

        with pytest.raises(
            ValueError, match="Unable to acquire a valid Microsoft Entra ID"
        ):
            refresh_openai_azure_ad_token(None)
