"""
    Return code of initialize call
"""
from enum import Enum

class ErrorInitialize(Enum):
    """Potential error during initialize."""

    SUCCESS = 0
    ERROR = 1
    NEED_TO_CLEAR_MTE = 11
