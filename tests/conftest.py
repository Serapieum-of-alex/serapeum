from typing import List
import pytest


@pytest.fixture(scope="module")
def manual_pdf() -> List[str]:
    return ["tests/data/test-pdf.pdf"]


@pytest.fixture(scope="module")
def num_manual_pages() -> int:
    return 40


@pytest.fixture(scope="module")
def model_id() -> str:
    return "google/gemma-2b-it"


def is_running_in_github_actions():
    """Check if the tests are running in GitHub Actions."""
    return os.getenv("GITHUB_ACTIONS") == "true"
