"""
    Data that will be send and received during the communication with client
    when learning or initializing the system.
"""
import json
from pykson import Pykson
import cv2
import numpy as np

from ML.Domain.LearningKnowledge import LearningKnowledge
from Domain.ErrorInitialize import ErrorInitialize
from Domain.VCLikeData import VCLikeData

class LearningData:
    """Data class and function."""

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
            "ml_validation" : None,
            "vc_like_data": None
        }

    def initialiaze_control_assist(self, id_ref, parameters):
        """Initialize keypoints and theirs descriptors from
           a dictionnary as well as the LearningKnowledge.
        """

        self.id_ref = id_ref
        self.mte_parameters = {
            "ratio" : parameters["ratio"],
            "size_small" : {
                "keypoints" : cv2.KeyPoint_convert(parameters["size_small"]["keypoints"]),
                "descriptors" : np.array(parameters["size_small"]["descriptors"], dtype=np.float32)
            },
            "size_medium" : {
                "keypoints" : cv2.KeyPoint_convert(parameters["size_medium"]["keypoints"]),
                "descriptors" : np.array(parameters["size_medium"]["descriptors"], dtype=np.float32)
            },
            "size_large" : {
                "keypoints" : cv2.KeyPoint_convert(parameters["size_large"]["keypoints"]),
                "descriptors" : np.array(parameters["size_large"]["descriptors"], dtype=np.float32)
            },
            "ml_validation" : Pykson().from_json(parameters["ml_validation"], LearningKnowledge),
            "vc_like_data": Pykson().from_json(parameters["vc_like_data"], VCLikeData)
        }

        return ErrorInitialize.SUCCESS.value

    def clean_control_assist(self, id_ref):
        """ Clear the memory from the given reference."""

        # If we try to clean the wrong reference
        if id_ref == self.id_ref:
            self.mte_parameters = None
            self.id_ref = None
            return 0
        else:
            return 1

    def fill_with_engine_for_learning(self, ratio, kp_small, desc_small,\
         kp_medium, desc_medium, kp_large, desc_large):
        """ Fill the class with informations that will be use
        to make comparison.
        """

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
            "ml_validation" : None,
            "vc_like_data": None
        }

    def fill_vc_like_learning_data(self, ratio, vc_like_data):
        """ Fill the class with informations that will be use
        to make comparison.
        """

        self.id_ref = -1
        self.mte_parameters["ratio"] = ratio
        self.mte_parameters["vc_like_data"] = vc_like_data

    def change_parameters_type_for_sending(self):
        """ Return a dictionnary containing the data after learning.
        Used to bypass JSON.dumps limitations.
        """
        if self.mte_parameters is None:
            return None
        else:
            to_send = {
                "ratio" : self.mte_parameters["ratio"],
                "ml_validation" : json.loads(Pykson().to_json(self.mte_parameters["ml_validation"]))
            }

            if self.mte_parameters["size_small"]["keypoints"] is not None :
                to_send["size_small"] = {
                    "keypoints" : cv2.KeyPoint_convert(self.mte_parameters["size_small"]["keypoints"])\
                        .tolist(),
                    "descriptors" : self.mte_parameters["size_small"]["descriptors"].tolist()
                }
            else:
                to_send["size_small"] = {
                    "keypoints" : [],
                    "descriptors" : []
                }

            if self.mte_parameters["size_medium"]["keypoints"] is not None :
                to_send["size_medium"] = {
                    "keypoints" : cv2.KeyPoint_convert(self.mte_parameters["size_medium"]["keypoints"])\
                        .tolist(),
                    "descriptors" : self.mte_parameters["size_medium"]["descriptors"].tolist()
                }
            else:
                to_send["size_medium"] = {
                    "keypoints" : [],
                    "descriptors" : []
                }

            if self.mte_parameters["size_large"]["keypoints"] is not None :
                to_send["size_large"] = {
                    "keypoints" : cv2.KeyPoint_convert(self.mte_parameters["size_large"]["keypoints"])\
                        .tolist(),
                    "descriptors" : self.mte_parameters["size_large"]["descriptors"].tolist()
                }
            else:
                to_send["size_large"] = {
                    "keypoints" : [],
                    "descriptors" : []
                }
            
            to_send["vc_like_data"] = json.loads(Pykson().to_json(self.mte_parameters["vc_like_data"]))

            return to_send

    def to_dict(self):
        """ Return all informations as a dictionnary.
        Used to bypass JSON.dumps limitations.
        """

        return {
            "id_ref" : self.id_ref,
            "mte_parameters" : self.mte_parameters
        }
