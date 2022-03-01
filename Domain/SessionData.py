import math

from SIFTEngine import SIFTEngine
from VCLikeEngine import VCLikeEngine

from Domain.LearningData import LearningData
from Domain.MTEAlgo import MTEAlgo
from MLValidation import MLValidation


class SessionData:
    def __init__(self, mte_algo, one_shot_mode, disable_histogram_matching, debug_mode, ransacount) -> None:
        self.rollback = 0
        self.orange_count_for_rollback = 0
        self.validation = 0
        self.resolution_change_allowed = 3
        self.reference = LearningData()
        self.mte_algo = mte_algo
        self.debug_mode = debug_mode

        self.format_resolution = None
        self.width_small = None
        self.width_medium = None
        self.width_large = None
        self.validation_width = None
        self.validation_height = None
        
        self.vc_like_engine = None
        self.sift_engine = None
        self.ml_validator = MLValidation()

        if self.mte_algo == MTEAlgo.VC_LIKE:
            self.vc_like_engine = VCLikeEngine(one_shot_mode=one_shot_mode, \
                disable_histogram_matching = disable_histogram_matching, debug_mode=self.debug_mode)
        else:
            self.sift_engine = SIFTEngine(maxRansac=ransacount)
        
        self.target = None

    def set_mte_parameters(self, ratio):
        """Edit values for globals parameters of the motion tracking engine."""

        if math.isclose(ratio, 16/9, rel_tol=1e-5):
            self.width_small = 400
            self.width_medium = 660
            self.width_large = 1730
            self.format_resolution = 16/9
        elif math.isclose(ratio, 4/3, rel_tol=1e-5):
            self.width_small = 350
            self.width_medium = 570
            self.width_large = 1730
            self.format_resolution = 4/3
        else:
            print("What kind of format is that ?")
            return False

        self.validation_width = self.width_small
        self.validation_height = int(self.validation_width*(1/self.format_resolution))
        if self.mte_algo in (MTEAlgo.SIFT_KNN, MTEAlgo.SIFT_RANSAC):
            self.sift_engine.set_parameters(self.width_small, self.width_medium,\
                                            self.width_large, self.format_resolution)
        elif self.mte_algo == MTEAlgo.VC_LIKE:
            self.vc_like_engine.set_parameters(self.format_resolution)

        return True
