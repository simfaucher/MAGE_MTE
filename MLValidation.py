
import os
import sys
import time
from copy import deepcopy
import json
import numpy as np
import cv2
from pykson import Pykson

from ML.Domain.LearningKnowledge import LearningKnowledge
from ML.Domain.Image import Image
from ML.Domain.ROIFeatureType import ROIFeatureType
from ML.Domain.ROIFeature import ROIFeature
from ML.Domain.ImageFilterType import ImageFilterType
from ML.Domain.Point2D import Point2D
from ML.Domain.ImageClass import ImageClass

from ML.LinesDetector import LinesDetector
from ML.BoxLearner import BoxLearner

CONFIG_SIGHTS_FILENAME = "learning_settings.json"

class MLValidation:
    def __init__(self):
        self.load_ml_settings()
        self.box_learner = BoxLearner(self.learning_settings.sights, \
            self.learning_settings.recognition_selector.uncertainty)

    def load_ml_settings(self):
        try:
            print("Reading the input file : {}".format(CONFIG_SIGHTS_FILENAME))
            with open(CONFIG_SIGHTS_FILENAME) as json_file:
                json_data = json.load(json_file)
        except IOError as error:
            sys.exit("The file {} doesn't exist.".format(CONFIG_SIGHTS_FILENAME))

        try:
            self.learning_settings = Pykson.from_json(json_data, LearningKnowledge, accept_unknown=True)
        except TypeError as error:
            sys.exit("Type error in {} with the attribute \"{}\". Expected {} but had {}.".format(error.args[0], error.args[1], error.args[2], error.args[3]))

    
    def learn(self, learning_data):
        if learning_data.mte_parameters["ml_validation"] is None:
            learning_data.mte_parameters["ml_validation"] = deepcopy(self.learning_settings)

            image_class = ImageClass()
            image_class.id = 0
            image_class.name = "Reference"

            h, w = learning_data.resized_image.shape[:2]

            for sight in learning_data.mte_parameters["ml_validation"].sights:
                pt_tl = Point2D()
                pt_tl.x = int(w / 2 - sight.width / 2)
                pt_tl.y = int(h / 2 - sight.height / 2)

                pt_br = Point2D()
                pt_br.x = pt_tl.x + sight.width
                pt_br.y = pt_tl.y + sight.height

                sight_image = learning_data.resized_image[pt_tl.y: pt_br.y, pt_tl.x: pt_br.x]
                # cv2.imshow("Sight", sight_image)

                for j, roi in enumerate(sight.roi):
                    image = Image()
                    image.sight_position = Point2D()
                    image.sight_position.x = pt_tl.x
                    image.sight_position.y = pt_tl.y
                    image.image_class = image_class

                    image_filter = ImageFilterType(roi.image_filter_type)

                    detector = LinesDetector(sight_image, image_filter)
                    mask = detector.detect()
                    # cv2.imshow("Sight mask", mask)

                    x = int(roi.x)
                    y = int(roi.y)
                    width = int(roi.width)
                    height = int(roi.height)

                    roi_mask = mask[y:y+height, x:x+width]
                    # cv2.imshow("ROI"+str(j), roi_mask)

                    # Feature extraction
                    feature_vector = roi.feature_type
                    vector = BoxLearner.extract_pixels_features(roi_mask, ROIFeatureType(feature_vector))

                    feature = ROIFeature()
                    feature.feature_type = ROIFeatureType(feature_vector)
                    feature.feature_vector = vector[0].tolist()

                    image.features.append(feature)

                    roi.images.append(image)

    def validate(self, learning_data, warped_image):
        success = len(learning_data.mte_parameters["ml_validation"].sights) > 0
        sum_distances = 0
        distances = []

        for sight in learning_data.mte_parameters["ml_validation"].sights:
            self.box_learner.get_knn_contexts(sight)
            self.box_learner.input_image = warped_image

            h, w = warped_image.shape[:2]

            pt_tl = Point2D()
            pt_tl.x = int(w / 2 - sight.width / 2)
            pt_tl.y = int(h / 2 - sight.height / 2)

            pt_br = Point2D()
            pt_br.x = pt_tl.x + sight.width
            pt_br.y = pt_tl.y + sight.height

            match = self.box_learner.find_target(pt_tl, pt_br)

            success = match.success if not match.success else success
            sum_distances += match.sum_distances
            distances += match.roi_distances

        return success, sum_distances, distances