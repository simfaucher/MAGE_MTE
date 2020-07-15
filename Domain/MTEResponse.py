from enum import Enum

class MTEResponse(Enum):
    TARGET_LOST = 1
    RED = 2
    ORANGE = 3
    GREEN = 4
    CAPTURE = 5
    