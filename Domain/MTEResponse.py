from enum import Enum

class MTEResponse(Enum):
    CAPTURE = 1
    GREEN = 2
    ORANGE = 3
    RED = 4
    TARGET_LOST = 5
    