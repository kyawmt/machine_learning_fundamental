"""
mlprep - Machine Learning Interview Review companion toolkit.

Provides standard dataset generators, shared matplotlib styling,
and automated exercise assertions.
"""

SITE = "http://localhost:5173/"

from mlprep.preprocessing import make_target_encoder

__all__ = ["SITE", "make_target_encoder"]
