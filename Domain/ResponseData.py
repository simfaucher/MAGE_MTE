from MTEResponse import MTEResponse

class ResponseData:
    def __init__(self, size:int, response: MTEResponse, translation_x: float, \
                 translation_y: float, direction: str, \
                 scale_w: float, scale_h: float):
        self.size = size
        self.response = response
        self.shift_x = translation_x
        self.shift_y = translation_y
        self.direction = direction
        self.scale_h = scale_h
        self.scale_w = scale_w
        