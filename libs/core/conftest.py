"""Core package conftest.

Shared pytest hooks re-exported by every provider conftest:
- ``--no-skip-doctest``: run doctest examples marked with ``+SKIP``
- ``--md-marker``: filter markdown code blocks by custom tags
"""

from __future__ import annotations

import doctest
import pathlib

import pytest

# ---------------------------------------------------------------------------
# Markdown code-block markers  (``python local``, ``python ci``, etc.)
# ---------------------------------------------------------------------------
_LANG_TAGS = frozenset({"py", "python", "python3"})
_BUILTIN_OPTIONS = frozenset({"notest", "continuation"})
_BUILTIN_PREFIXES = ("fixture:", "runner:", "retry:")
_tag_cache: dict[pathlib.Path, dict[int, set[str]]] = {}


def _extract_md_tags(filepath: pathlib.Path) -> dict[int, set[str]]:
    """Return ``{start_line: {tag, …}}`` for every Python fence in *filepath*."""
    from markdown_it import MarkdownIt

    parser = MarkdownIt("commonmark")
    tokens = parser.parse(filepath.read_text(encoding="utf-8"))

    result: dict[int, set[str]] = {}
    for block in tokens:
        if block.type != "fence" or not block.map:
            continue

        info = block.info.split()
        lang = info[0] if info else None
        if lang not in _LANG_TAGS:
            continue

        tags = {
            o
            for o in info[1:]
            if o not in _BUILTIN_OPTIONS
            and not any(o.startswith(p) for p in _BUILTIN_PREFIXES)
        }
        if tags:
            # Same arithmetic the plugin uses: start_line = offset + map[0] + 1
            # For .md files the offset is 0.
            result[block.map[0] + 1] = tags

    return result


def _get_tags(filepath: pathlib.Path) -> dict[int, set[str]]:
    if filepath not in _tag_cache:
        _tag_cache[filepath] = _extract_md_tags(filepath)
    return _tag_cache[filepath]


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    """Register shared CLI flags."""
    parser.addoption(
        "--no-skip-doctest",
        action="store_true",
        default=False,
        help="Ignore all '# doctest: +SKIP' directives so skipped examples run.",
    )
    parser.addoption(
        "--md-marker",
        action="append",
        default=[],
        help=(
            "Only run markdown code blocks tagged with this marker. "
            "Repeatable: --md-marker local --md-marker ci runs both."
        ),
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item],
) -> None:
    """Post-collection hooks: --no-skip-doctest and --md-marker filtering."""
    # -- doctest SKIP removal ------------------------------------------------
    if config.getoption("--no-skip-doctest"):
        for item in items:
            if isinstance(item, pytest.DoctestItem):
                for example in item.dtest.examples:
                    example.options.pop(doctest.SKIP, None)

    # -- markdown code-block marker filtering --------------------------------
    try:
        from pytest_markdown_docs.plugin import MarkdownInlinePythonItem
    except ImportError:
        return

    md_markers: list[str] = config.getoption("--md-marker", default=[])

    # Phase 1 — stamp every markdown-docs item with its fence tags as markers.
    for item in items:
        if not isinstance(item, MarkdownInlinePythonItem):
            continue
        src = item.test_definition.source_path
        if src.suffix not in (".md", ".mdx", ".svx"):
            continue
        for tag in _get_tags(src).get(item.start_line, set()):
            item.add_marker(tag)

    # Phase 2 — if --md-marker was given, keep only matching markdown items.
    if md_markers:
        wanted = set(md_markers)
        selected: list[pytest.Item] = []
        deselected: list[pytest.Item] = []

        for item in items:
            if isinstance(item, MarkdownInlinePythonItem):
                own = {m.name for m in item.iter_markers()}
                if own & wanted:
                    selected.append(item)
                else:
                    deselected.append(item)
            else:
                # Non-markdown items are always kept.
                selected.append(item)

        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = selected
