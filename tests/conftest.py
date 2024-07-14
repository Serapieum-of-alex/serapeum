from typing import List
import pytest


@pytest.fixture(scope="module")
def manual_pdf() -> List[str]:
    return ["tests/data/test-pdf.pdf"]


@pytest.fixture(scope="module")
def num_manual_pages() -> int:
    return 40
