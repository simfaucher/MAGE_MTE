"""
    Response type for client
"""
from Domain.MTEResponse import MTEResponse

class ResponseData:
    """This will be the response send to the client"""

    def __init__(self, size: [int, int], response: MTEResponse, translation_x: float, \
                 translation_y: float, direction: str, \
                 scale_w: float, scale_h: float, status=None):
        self.requested_image_size = size
        self.flag = response.name
        self.target_data = {
            "translations" : (translation_x, translation_y),
            "scales" : (scale_h, scale_w)
        }
        self.user_information = direction
        self.status = status

    def set_status(self, status):
        self.status = status
