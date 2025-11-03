"""Top-level package for TQQQ machine learning trend framework."""

from importlib.metadata import version, PackageNotFoundError

__all__ = ["get_version"]


def get_version() -> str:
    """Return the installed package version.

    Returns
    -------
    str
        Installed version if available, otherwise ``"0.0.0"``.
    """
    try:
        return version("tqqq-ml-trend")
    except PackageNotFoundError:
        return "0.0.0"
