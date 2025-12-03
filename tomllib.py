"""Compatibility shim for Python < 3.11 where tomllib is unavailable."""

from tomli import *  # noqa: F401,F403
