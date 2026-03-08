"""Core package conftest."""

from __future__ import annotations

import doctest

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the --no-skip-doctest CLI flag."""
    parser.addoption(
        "--no-skip-doctest",
        action="store_true",
        default=False,
        help="Ignore all '# doctest: +SKIP' directives so skipped examples run.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item],
) -> None:
    """Strip SKIP from every doctest example when --no-skip-doctest is active."""
    if not config.getoption("--no-skip-doctest"):
        return

    for item in items:
        if isinstance(item, pytest.DoctestItem):
            for example in item.dtest.examples:
                example.options.pop(doctest.SKIP, None)
