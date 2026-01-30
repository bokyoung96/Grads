"""Compatibility shim: export KR root constants at top-level.

This preserves existing imports `from root import ...` while the real
implementation lives under KR/root.py.
"""

from KR.root import *  # noqa: F401,F403
