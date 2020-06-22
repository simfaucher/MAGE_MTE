"""
    Response type for client
"""
from Domain.MTEResponse import MTEResponse

class ResponseData:
    """This will be the response send to the client"""

    def __init__(self, size: int, response: MTEResponse, translation_x: float, \
                 translation_y: float, direction: str, \
                 scale_w: float, scale_h: float):
        self.size = size
        self.response = response
        self.shift_x = translation_x
        self.shift_y = translation_y
        self.direction = direction
        self.scale_h = scale_h
        self.scale_w = scale_w

    def convert_to_dict(self):
        """Convert the class to a dictonnary to allow
        a json transform.
        """

        return {"size" : self.size,\
                "response" : self.response.name,\
                "shift_x" : self.shift_x,\
                "shift_y" : self.shift_y,\
                "direction" : self.direction,\
                "scale_h" : self.scale_h,
                "scale_w" : self.scale_w
                }
