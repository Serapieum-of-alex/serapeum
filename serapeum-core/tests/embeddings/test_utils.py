from serapeum.core.embeddings.mock_embed_model import MockEmbedding
from serapeum.core.embeddings.utils import resolve_embed_model
from pytest import MonkeyPatch


def test_resolve_embed_model(monkeypatch: MonkeyPatch) -> None:
    # Test None
    embed_model = resolve_embed_model(None)
    assert isinstance(embed_model, MockEmbedding)
