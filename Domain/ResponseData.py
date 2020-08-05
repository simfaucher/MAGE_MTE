"""
    Response type for client
"""
from Domain.MTEResponse import MTEResponse
from Domain.UserInformation import UserInformation
from Domain.ErrorRecognition import ErrorRecognition

class ResponseData:
    """This will be the response send to the client"""

    def __init__(self, size: [int, int], response: MTEResponse, translation_x: float, \
                 translation_y: float, direction: UserInformation, \
                 scale_w: float, scale_h: float, status=None):
        self.requested_image_size = size
        self.flag = response
        self.target_data = {
            "translations" : (translation_x, translation_y),
            "scales" : (scale_h, scale_w)
        }
        if direction is None:
            self.user_information = UserInformation.CENTERED
        else:
            self.user_information = direction
        if status is not None:
            self.status = status
        else:
            self.status = ErrorRecognition.ERROR

    def set_status(self, status):
        """Status setter.
        Needed to choose precisely what kind of Error we have
        """

        self.status = status

    def to_dict(self):
        """ Convert the response to a dictionnary to
        bypass JSON limitations.
        """

        return {
            "requested_image_size" : self.requested_image_size,
            "flag" : self.flag.value,
            "target_data" : {
                "translations" : self.target_data["translations"],
                "scales" : self.target_data["scales"]
            },
            "user_information" : self.user_information.value,
            "status" : self.status.value
        }
