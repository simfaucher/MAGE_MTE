from Domain.SiftData import SiftData

class LearningData:
    def __init__(self, num: int, label: str, resized_image, full_image):
        self.id = num
        self.label = label
        self.resized_image = resized_image
        self.full_image = full_image

        self.sift_data = None
        self.ml_data = None
