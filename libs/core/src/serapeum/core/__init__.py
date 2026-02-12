"""serapeum-core."""

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as get_version
except ImportError:  # pragma: no cover
    from importlib_metadata import (  # type: ignore[import-not-found]
        PackageNotFoundError,
    )
    from importlib_metadata import (
        version as get_version,  # type: ignore[import-not-found]
    )


try:
    __version__ = get_version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__doc__ = """
serapeum - serapeum utility package
"""
