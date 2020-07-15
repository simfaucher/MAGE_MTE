import sys
import glob
import itertools
from copy import deepcopy
import json
from pykson import Pykson
import numpy as np
import cv2
import imutils
from imutils.video import FPS
import matplotlib.pyplot as plt

from ML.Domain.LearningKnowledge import LearningKnowledge
from ML.Domain.Image import Image
from ML.Domain.ROIFeatureType import ROIFeatureType
from ML.Domain.ROIFeature import ROIFeature
from ML.Domain.ImageFilterType import ImageFilterType
from ML.Domain.Point2D import Point2D
from ML.Domain.ImageClass import ImageClass

from ML.LinesDetector import LinesDetector
from ML.BoxLearner import BoxLearner

from Domain.LearningData import LearningData
from Domain.VCLikeData import VCLikeData

CONFIG_VALIDATION_SIGHTS_FILENAME = "learning_settings_validation.json"
CONFIG_VALIDATION_SIGHTS_2_FILENAME = "learning_settings_validation2.json"
ROTATION_IMAGES_FOLDER = "images/rotation/*"

class VCLikeEngine:
    def __init__(self):
        self.learning_settings = self.load_ml_settings(CONFIG_VALIDATION_SIGHTS_FILENAME)
        self.box_learner = BoxLearner(self.learning_settings.sights, \
            self.learning_settings.recognition_selector.uncertainty)

        self.learning_settings2 = self.load_ml_settings(CONFIG_VALIDATION_SIGHTS_2_FILENAME)
        self.box_learner2 = BoxLearner(self.learning_settings2.sights, \
            self.learning_settings2.recognition_selector.uncertainty)

        self.last_position_found = None

    def learn(self, learning_data):
        if learning_data.vc_like_data is None:
            dataset = self.generate_dataset(learning_data.resized_image)
            learning_settings = self.learn_ml_data(dataset)
            # learning_settings2 = self.learn_ml_data2(dataset)
            learning_settings2 = {}

            learning_data.vc_like_data = VCLikeData(learning_settings, learning_settings2)

    def generate_dataset(self, image):
        h, w = image.shape[:2]

        dataset = []
        # Scale levels
        # s1 = [0.5, 0.75, 0.85]
        # s2 = np.arange(0.9, 1.1, 0.01)
        # s3 = [1.15, 1.25, 1.5]
        # scales = itertools.chain(s1, s2, s3)
        scales = np.arange(0.9, 1.1, 0.05)
        for scale in scales:
            scale = round(scale, 2)

            # Rotation levels
            # a1 = range(-40, -11, 10)
            # a2 = range(-10, 11, 1)
            # a3 = range(20, 41, 10)
            # angles = itertools.chain(a1, a2, a3)
            angles = [0, ]
            for angle in angles:
                # scaled = self.scale_image(image, scale)
                M = cv2.getRotationMatrix2D(((w-1)/2.0, (h-1)/2.0), angle, scale)
                transformed = cv2.warpAffine(image, M, (w, h))

                dataset.append(({"scale": scale, "angle": angle}, transformed))

                # ht, wt = transformed.shape[:2]
                # cv2.imshow("Scale:{}, rotation:{}, width{}:, height:{}".format(scale, angle, ht, wt), transformed)
                # cv2.waitKey(0)

        return dataset

    def load_ml_settings(self, filename):
        try:
            print("Reading the input file : {}".format(filename))
            with open(filename) as json_file:
                json_data = json.load(json_file)
        except IOError as error:
            sys.exit("The file {} doesn't exist.".format(filename))

        try:
            return Pykson.from_json(json_data, LearningKnowledge, accept_unknown=True)
        except TypeError as error:
            sys.exit("Type error in {} with the attribute \"{}\". Expected {} but had {}.".format(error.args[0], error.args[1], error.args[2], error.args[3]))

    def learn_ml_data(self, dataset):
        learning_settings = deepcopy(self.learning_settings)

        # for sight in learning_settings.sights:
        #     for j, roi in enumerate(sight.roi):
        #         roi.images = []

        for i, data in enumerate(dataset):
            attr, image = data[:]
            class_id = int(attr["scale"]*100)
            class_name = "scale: {}".format(attr["scale"])

            self.learn_image(learning_settings, class_id, class_name, image)

        return learning_settings

    def learn_ml_data2(self, dataset, scale=100):
        learning_settings2 = deepcopy(self.learning_settings2)

        # for sight in self.learning_settings2.sights:
        #     for j, roi in enumerate(sight.roi):
        #         roi.images = []

        for i, data in enumerate(dataset):
            attr, image = data[:]

            current_scale = int(attr["scale"]*100)

            if current_scale == scale:
                class_id = int(attr["angle"]) + 100
                class_name = "angle: {}".format(attr["angle"])

                self.learn_image(learning_settings2, class_id, class_name, image)

        return learning_settings2

    def learn_image(self, learning_settings, class_id, class_name, img):
        # Learn ML data

        image_class = ImageClass()
        image_class.id = class_id
        image_class.name = class_name

        h, w = img.shape[:2]

        for sight in learning_settings.sights:
            pt_tl = Point2D()
            pt_tl.x = int(w / 2 - sight.width / 2)
            pt_tl.y = int(h / 2 - sight.height / 2)

            pt_br = Point2D()
            pt_br.x = pt_tl.x + sight.width
            pt_br.y = pt_tl.y + sight.height

            sight_image = img[pt_tl.y: pt_br.y, pt_tl.x: pt_br.x]
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

    def ml_validation(self, learning_settings, box_learner, image):
        success = len(learning_settings.sights) > 0

        matches = []

        for sight in learning_settings.sights:
            box_learner.get_knn_contexts(sight)
            # box_learner.input_image = image

            # h, w = image.shape[:2]

            # pt_tl = Point2D()
            # pt_tl.x = int(w / 2 - sight.width / 2)
            # pt_tl.y = int(h / 2 - sight.height / 2)

            # pt_br = Point2D()
            # pt_br.x = pt_tl.x + sight.width
            # pt_br.y = pt_tl.y + sight.height

            # match = box_learner.find_target(pt_tl, pt_br)
            match = box_learner.scan(image)

            success = match.success if not match.success else success
            matches.append(match)

        return success, matches

    # def scale_image(self, image, scale):
    #     if scale == 1:
    #         return image

    #     h, w, c = image.shape

    #     resized = cv2.resize(image, None, fx=scale, fy=scale)
    #     h_r, w_r = resized.shape[:2]

    #     margin_h = int(abs(h_r - h) / 2)
    #     margin_w = int(abs(w_r - w) / 2)
    #     if scale < 1:
    #         dest = np.zeros((h, w, c), dtype=image.dtype)
    #         dest[margin_h: margin_h + h_r, margin_w: margin_w + w_r] = resized
    #     else:
    #         dest = resized[margin_h: margin_h + h, margin_w: margin_w + w]

    #     return dest

    def find_target(self, image, learning_data):
        #TODO: scan -> erreur sur le box learner 2
        h, w = image.shape[:2]

        # Scale
        learning_settings = learning_data.vc_like_data.learning_settings
        # success1, matches1 = self.ml_validation(learning_settings, self.box_learner, image)

        self.box_learner.get_knn_contexts(learning_settings.sights[0])

        match1 = None

        if self.last_position_found is not None:
            match1 = self.box_learner.optimised_scan(image, anchor_point=self.last_position_found)

        # If never found a position or the optimised scan failed
        if match1 is None or not match1.success:
            match1 = self.box_learner.scan(image)

        success1 = match1.success
        # success2 = False

        scale = 1
        angle = 0
        translation = (0, 0)

        if success1:
            # scale = 1 + float(100 - matches1[0].predicted_class)/100
            scale = 1 / (match1.predicted_class / 100)
            print("Scale success, class: {}, distance: {}".format(match1.predicted_class, match1.roi_distances[0]))

            translation = (match1.anchor.x - learning_settings.sights[0].anchor.x, \
                match1.anchor.y - learning_settings.sights[0].anchor.y)

            self.last_position_found = match1.anchor

            # Data for plotting

            # M = cv2.getRotationMatrix2D(((w-1)/2.0, (h-1)/2.0), 0, scale)
            # scaled = cv2.warpAffine(image, M, (w, h))

            # # scaled = self.scale_image(image, scale)

            # # cv2.imshow("Scaled", scaled)

            # # Rotation
            # learning_settings2 = learning_data.vc_like_data.learning_settings2
            # # success2, matches2 = self.ml_validation(learning_settings2, self.box_learner2, scaled)
            # self.box_learner2.get_knn_contexts(learning_settings2.sights[0])
            # match2 = self.box_learner2.optimised_scan(image, anchor_point=match1.anchor)

            # if not match2.success:
            #     match2 = self.box_learner2.scan(image)

            # success2 = match2.success

            # if success2:
            #     angle = -1 * (match2.predicted_class - 100)

            #     # Save this position for next turn
            #     self.last_position_found = match2.anchor

            #     # Data for plotting

            #     print("Rotation success, class: {}".format(match2.predicted_class - 100))
            # else:
            #     print("Rotation fail")
        else:
            print("Scale fail")

        M = cv2.getRotationMatrix2D(((w-1)/2.0, (h-1)/2.0), angle, scale)
        transformed = cv2.warpAffine(image, M, (w, h))

        # cv2.imshow("Image", image)
        # cv2.imshow("Transformed", transformed)
        # cv2.waitKey(100)

        # return (success1 and success2), (scale, scale), (angle, angle), transformed
        return success1, (scale, scale), (angle, angle), translation, transformed

if __name__ == "__main__":
    image_ref = cv2.imread("videos/T1.1/vlcsnap-2020-03-02-15h59m47s327.png")
    image_ref = imutils.resize(image_ref, width=640)

    app = VCLikeEngine()
    app.learn(image_ref)

    # for image_path in glob.glob(ROTATION_IMAGES_FOLDER):
    #     image = cv2.imread(image_path)
    #     image = imutils.resize(image, width=640)
    #     success, matches = app.ml_validation(image)

    #     if success:
    #         print("{}, success, class: {}", image_path, matches[0].predicted_class)
    #     else:
    #        print("{}, fail")

    #     cv2.imshow(image_path, image)
    #     cv2.waitKey(0)

    image = cv2.imread("videos/T1.1/vlcsnap-2020-03-09-11h25m42s248.png")
    image = imutils.resize(image, width=640)
    dataset = app.generate_dataset(image)

    for data, image in dataset:
        success, scale, angle, transformed = app.find_target(image)

        cv2.imshow("Image", image)
        cv2.imshow("Transformed", transformed)
        cv2.waitKey(100)
