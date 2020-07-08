from enum import Enum

class MTEMode(Enum):
    NEUTRAL = 0
    VALIDATION_REFERENCE = 1
    INITIALIZE_MTE = 2
    MOTION_TRACKING = 3
    CLEAR_MTE = 4
