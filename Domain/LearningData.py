from Domain.SiftData import SiftData
import cv2 as cv2

class LearningData:
    def __init__(self):
        self.id_ref = None
        self.mte_parameters = {
            "ratio" : None,
            "size_small" : {
                "keypoints" : None,
                "descriptors" : None
            },
            "size_medium" : {
                "keypoints" : None,
                "descriptors" : None
            },
            "size_large" : {
                "keypoints" : None,
                "descriptors" : None
            },
            "ml_validation" : None
        }

    def initialiaze_control_assist(self, id_ref, parameters):
        self.id_ref = id_ref
        self.mte_parameters = {
            "ratio" : parameters["ratio"],
            "size_small" : {
                "keypoints" : parameters["size_small"]["kp_small"],
                "descriptors" : parameters["size_small"]["desc_small"]
            },
            "size_medium" : {
                "keypoints" : parameters["size_medium"]["kp_medium"],
                "descriptors" : parameters["size_medium"]["desc_medium"]
            },
            "size_large" : {
                "keypoints" : parameters["size_large"]["kp_large"],
                "descriptors" : parameters["size_large"]["desc_large"]
            },
            "ml_validation" : parameters["ml_validation"]
        }
        return 0

    def clean_control_assist(self, id_ref):
        # If we try to clean the wrong reference
        if id_ref == self.id_ref:
            self.mte_parameters = None
            self.id_ref = None
            return 0
        else:
            return 1

    def fill_with_engine_for_learning(self, ratio, kp_small, desc_small, kp_medium, desc_medium, kp_large, desc_large):
        self.id_ref = -1
        self.mte_parameters = {
            "ratio" : ratio,
            "size_small" : {
                "keypoints" : kp_small,
                "descriptors" : desc_small
            },
            "size_medium" : {
                "keypoints" : kp_medium,
                "descriptors" : desc_medium
            },
            "size_large" : {
                "keypoints" : kp_large,
                "descriptors" : desc_large
            },
            "ml_validation" : None
        }

    def change_parameters_type_for_sending(self):
        return {
            "ratio" : self.mte_parameters["ratio"],
            "size_small" : {
                "keypoints" : cv2.KeyPoint_convert(self.mte_parameters["size_small"]["keypoints"]).tolist(),
                "descriptors" : self.mte_parameters["size_small"]["descriptors"].tolist()
            },
            "size_medium" : {
                "keypoints" : cv2.KeyPoint_convert(self.mte_parameters["size_medium"]["keypoints"]).tolist(),
                "descriptors" : self.mte_parameters["size_medium"]["descriptors"].tolist()
            },
            "size_large" : {
                "keypoints" : cv2.KeyPoint_convert(self.mte_parameters["size_large"]["keypoints"]).tolist(),
                "descriptors" : self.mte_parameters["size_large"]["descriptors"].tolist()
            },
            "ml_validation" : self.mte_parameters["ml_validation"]
        }
