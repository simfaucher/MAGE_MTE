"""
    Return code for learning call.
"""
from enum import Enum

class ErrorLearning(Enum):
    """Potential error when learning a reference."""

    SUCCESS = 0
    ERROR = 11
    INVALID_FORMAT = 12
    ERROR_REFERENCE_IS_BLURRED = 12
    GAUSSIAN_BLUR_FAILURE = 21
    VERTICAL_BLUR_FAILURE = 22
    HORIZONTAL_BLUR_FAILURE = 23
