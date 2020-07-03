from Domain.SiftData import SiftData

class LearningData:
    def __init__(self):
        self.id_ref = None
        self.mte_paramters = {
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
        self.mte_paramters = {
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
            self.mte_paramters = None
            self.id_ref = None
            return 0
        else:
            return 1
