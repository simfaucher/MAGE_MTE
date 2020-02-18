from Domain.SiftData import SiftData

class LearningData:
    def __init__(self, num: int, label: str, image_640, full_image):
        self.id = num
        self.label = label
        self.image_640 = image_640
        self.full_image = full_image

        self.sift_data = None
        self.ml_data = None
