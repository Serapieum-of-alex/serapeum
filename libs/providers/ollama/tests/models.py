"""Test data models for Ollama integration tests."""
from __future__ import annotations
import os

import ollama as ollama_sdk


# created clients in this test file are used in the pytest.mark.skipif decorator and can not be made as fixtures
# so keep them as they are.
# currently there are no embedding models in ollama cloud

api_key: str = os.environ.get("OLLAMA_API_KEY")


# Local client (used by existing tests)
try:
    client = ollama_sdk.Client()        # type: ignore
    client.list()
except Exception:
    client = None


# Cloud client (used by cloud tests)
try:
    cloud_client = ollama_sdk.Client(       # type: ignore
        host="https://api.ollama.com",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    cloud_client.list()   # connectivity probe — raises if unreachable / unauthorized
except Exception:
    cloud_client = None


# In CI use the cloud client so local-gated tests run against the cloud backend.
# Locally fall back to the local Ollama server.
client = cloud_client if (is_ci and cloud_client is not None) else _local_client

