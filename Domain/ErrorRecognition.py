"""
    Return code of recognition call
"""
from enum import Enum

class ErrorRecognition(Enum):
    SUCCESS = 0
    # GREEN
    IMAGE_IS_BLURRED = 201
    NOT_CENTERED_WITH_TARGET = 202
    # ORANGE
    NOT_ENOUGHT_MATCH = 301
    MEANS_OUT_OF_LIMITS = 302
    ABERRATION_VALUE = 303
    # RED
    TARGET_LOST = 401
    MISMATCH_SIZE_WITH_REF = 402
    NOT_ENOUGHT_KEYPOINTS = 403
    NOT_ENOUGHT_MATCH_CRITICAL = 404
