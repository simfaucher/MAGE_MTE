"""
    Information to indicate the direction to go for the client's user
"""
from enum import Enum

class UserInformation(Enum):
    """ Mode available for the engine.
    Neutral is only available for client.
    """

    CENTERED = 0
    UP = 1
    UP_RIGHT = 2
    RIGHT = 3
    DOWN_RIGHT = 4
    DOWN = 5
    DOWN_LEFT = 6
    LEFT = 7
    UP_LEFT = 8

    BIG_UP = 11
    BIG_RIGHT = 13
    BIG_DOWN = 15
    BIG_LEFT = 17
