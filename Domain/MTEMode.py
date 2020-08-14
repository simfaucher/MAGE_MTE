"""
    Mode for Client and Server
"""
from enum import Enum

class MTEMode(Enum):
    """ Mode available for the engine.
    Neutral is only available for client.
    """

    NEUTRAL = 0
    VALIDATION_REFERENCE = 1
    INITIALIZE_MTE = 2
    MOTION_TRACKING = 3
    CLEAR_MTE = 4
    RUNNING_VERIFICATION = 5
