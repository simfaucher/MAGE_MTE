import numpy as np

class RecognitionData:
    def __init__(self, success: bool, recog_ret_data: dict, nb_kp: int,
                 nb_match: int, sum_translation: float, \
                 sum_skew: float, sum_distances: float, \
                 dist_roi: list, warped_img: np.ndarray, \
                 scales: tuple, translations: tuple):
        self.success = success
        self.recog_ret_data = recog_ret_data
        self.nb_kp = nb_kp
        self.nb_match = nb_match
        self.sum_translation = sum_translation
        self.sum_skew = sum_skew
        self.sum_distances = sum_distances
        self.dist_roi = dist_roi
        self.warped_img = warped_img
        self.scales = scales
        self.translations = translations
        