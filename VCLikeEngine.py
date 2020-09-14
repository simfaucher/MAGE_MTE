import sys
import time
import glob
import math
import itertools
from copy import deepcopy
import json
from pykson import Pykson
import numpy as np
import cv2
import imutils
from imutils.video import FPS
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt

from ML.Domain.LearningKnowledge import LearningKnowledge
from ML.Domain.Image import Image
from ML.Domain.ROIFeatureType import ROIFeatureType
from ML.Domain.ROIFeature import ROIFeature
from ML.Domain.ImageFilterType import ImageFilterType
from ML.Domain.Point2D import Point2D
from ML.Domain.ImageClass import ImageClass
from ML.Domain.IndexedLearningKnowledge import IndexedLearningKnowledge
from ML.Domain.LearnerMatch import LearnerMatch

from ML.LinesDetector import LinesDetector
from ML.BoxLearner import BoxLearner

from Domain.LearningData import LearningData
from Domain.VCLikeData import VCLikeData
from Domain.HistogramMatchingData import HistogramMatchingData
from Domain.MTEResponse import MTEResponse

LEARNING_SETTINGS_85 = "learning_settings_85.json"
LEARNING_SETTINGS_64 = "learning_settings_64.json"

TIMEOUT_LIMIT_SEC = 7

class VCLikeEngine:
    def __init__(self, one_shot_mode=False, disable_histogram_matching=False, debug_mode=False):
        self.reference_image = None

        self.last_match = None
        self.last_response_type = MTEResponse.ORANGE
        self.last_translation = (0, 0)
        self.mode = 0
        self.scale = 100
        self.learning_settings_85 = self.load_ml_settings(LEARNING_SETTINGS_85)
        self.learning_settings_64 = self.load_ml_settings(LEARNING_SETTINGS_64)
        self.validation_count = 0
        self.disable_histogram_matching = disable_histogram_matching

        self.ratio = None

        # Box learners
        self.box_learners_85_singlescale = {}
        self.box_learners_64_singlescale = {}
        self.box_learner_85_multiscale = None
        self.box_learner_64_multiscale = None

        # Histogram matching data
        self.histogram_matching_data = None

        # Parameters
        self.image_width = 176
        self.image_height = 97
        
        self.nb_frames = 0
        self.one_shot_mode = one_shot_mode

        self.nb_following_captures = 1

        self.debug_mode = debug_mode
        # self.to_skip = 1/2

    # def learn(self, image, learning_data):
    #     if learning_data.mte_parameters["vc_like_data"] is None:
    #         dataset = self.generate_dataset(image)
    #         learning_settings = self.learn_ml_data(dataset)
    #         # learning_settings2 = self.learn_ml_data2(dataset)
    #         learning_settings2 = {}

    #         vc_like_data = VCLikeData()
    #         vc_like_data.learning_settings = learning_settings
    #         vc_like_data.learning_settings2 = learning_settings2

    #         learning_data.fill_vc_like_learning_data(self.ratio, vc_like_data)
    
    def set_parameters(self, ratio):
        self.ratio = ratio

    def load_data(self, json_data):
        try:
            return Pykson.from_json(json_data, LearningKnowledge, accept_unknown=True)
        except TypeError as error:
            sys.exit("Type error in {} with the attribute \"{}\". Expected {} but had {}.".format(error.args[0], error.args[1], error.args[2], error.args[3]))

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
    
    def generate_dataset(self, image):
        h, w = image.shape[:2]

        dataset = []
        # Scale levels
        scales = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
        for scale in scales:
            scale = round(scale, 2)

            # Rotation levels
            angles = [0, ]
            for angle in angles:
                M = cv2.getRotationMatrix2D(((w-1)/2.0, (h-1)/2.0), angle, scale)
                transformed = cv2.warpAffine(image, M, (w, h))

                dataset.append(({"scale": scale, "angle": angle}, transformed))

                # ht, wt = transformed.shape[:2] # Debug
                # cv2.imshow("Scale:{}, rotation:{}, width{}:, height:{}".format(scale, angle, ht, wt), transformed) # Debug
                # cv2.waitKey(0) # Debug

        return dataset

    def learn(self, input_image, learning_data):
        if learning_data.mte_parameters["vc_like_data"] is None:
            vc_like_data = VCLikeData()
            image = cv2.resize(input_image, (self.image_width, self.image_height))

            self.reference_image = image #TODO: remove ?

            # Generate data for histogram matching
            vc_like_data.histogram_matching_data = self.generate_histogram_data(image)

            # Generate dataset
            dataset = self.generate_dataset(image)

            # 85x48 learnings
            # learning_settings_85_singlescale = {}
            # self.box_learners_85_singlescale = {}
            vc_like_data.learning_settings_85_multiscale = deepcopy(self.learning_settings_85)

            # 64x64 learnings
            # learning_settings_64_singlescale = {}
            # self.box_learners_64_singlescale = {}
            vc_like_data.learning_settings_64_multiscale = deepcopy(self.learning_settings_64)

            for i, data in enumerate(dataset):
                attr, image = data[:]

                scale = int(attr["scale"]*100)

                class_id = scale
                class_name = "scale: {}".format(scale)

                # Create every 85x48 single-scale box learners
                learning_settings_singlescale_85 = deepcopy(self.learning_settings_85)
                self.learn_image(learning_settings_singlescale_85, class_id, class_name, image)

                learning_settings_singlescale_85_indexed = IndexedLearningKnowledge()
                learning_settings_singlescale_85_indexed.scale = scale
                learning_settings_singlescale_85_indexed.learning_knowledge = learning_settings_singlescale_85
                vc_like_data.learning_settings_85_singlescale.append(learning_settings_singlescale_85_indexed)

                # box_learner_singlescale_85 = BoxLearner(learning_settings_singlescale_85.sights, 0)
                # box_learner_singlescale_85.get_knn_contexts(learning_settings_singlescale_85.sights[0])

                # self.box_learners_85_singlescale[scale] = box_learner_singlescale_85

                # Add learning to the 85x48 multi-scale learning
                self.learn_image(vc_like_data.learning_settings_85_multiscale, class_id, class_name, image)

                # Create every 64x64 single-scale box learners
                learning_settings_singlescale_64 = deepcopy(self.learning_settings_64)
                self.learn_image(learning_settings_singlescale_64, class_id, class_name, image)
                
                learning_settings_singlescale_64_indexed = IndexedLearningKnowledge()
                learning_settings_singlescale_64_indexed.scale = scale
                learning_settings_singlescale_64_indexed.learning_knowledge = learning_settings_singlescale_64
                vc_like_data.learning_settings_64_singlescale.append(learning_settings_singlescale_64_indexed)
                
                # box_learner_singlescale_64 = BoxLearner(learning_settings_singlescale_64.sights, 0)
                # box_learner_singlescale_64.get_knn_contexts(learning_settings_singlescale_64.sights[0])

                # self.box_learners_64_singlescale[scale] = box_learner_singlescale_64

                # # Add learning to the 64x64 multi-scale learning
                self.learn_image(vc_like_data.learning_settings_64_multiscale, class_id, class_name, image)
            
            # Create 85x48 multi-scale box learner
            # self.box_learner_85_multiscale = BoxLearner(learning_settings_85_multiscale.sights, 0)
            # self.box_learner_85_multiscale.get_knn_contexts(learning_settings_85_multiscale.sights[0])

            # Create 64x64 multi-scale box learner
            # self.box_learner_64_multiscale = BoxLearner(learning_settings_64_multiscale.sights, 0)
            # self.box_learner_64_multiscale.get_knn_contexts(learning_settings_64_multiscale.sights[0])
        
            learning_data.fill_vc_like_learning_data(self.ratio, vc_like_data)
    
    def init_engine(self, learning_data):
        self.mode = 0
        self.scale = 100
        self.validation_count = 0
        # Create every 85x48 single-scale box learners
        learning_settings_85_singlescale_indexed = learning_data.mte_parameters["vc_like_data"].learning_settings_85_singlescale

        for ls_indexed in learning_settings_85_singlescale_indexed:
            box_learner_singlescale = BoxLearner(ls_indexed.learning_knowledge.sights, 0)
            box_learner_singlescale.get_knn_contexts(ls_indexed.learning_knowledge.sights[0])

            self.box_learners_85_singlescale[ls_indexed.scale] = box_learner_singlescale

        # Create every 64x64 single-scale box learners
        learning_settings_64_singlescale_indexed = learning_data.mte_parameters["vc_like_data"].learning_settings_64_singlescale

        for ls_indexed in learning_settings_64_singlescale_indexed:
            box_learner_singlescale = BoxLearner(ls_indexed.learning_knowledge.sights, 0, debug_mode=self.debug_mode)
            box_learner_singlescale.get_knn_contexts(ls_indexed.learning_knowledge.sights[0])

            self.box_learners_64_singlescale[ls_indexed.scale] = box_learner_singlescale

        # Create 85x48 multi-scale box learner
        self.box_learner_85_multiscale = BoxLearner(learning_data.mte_parameters["vc_like_data"].learning_settings_85_multiscale.sights, 0)
        self.box_learner_85_multiscale.get_knn_contexts(learning_data.mte_parameters["vc_like_data"].learning_settings_85_multiscale.sights[0])

        # Create 64x64 multi-scale box learner
        self.box_learner_64_multiscale = BoxLearner(learning_data.mte_parameters["vc_like_data"].learning_settings_64_multiscale.sights, 0, \
            debug_mode=self.debug_mode)
        self.box_learner_64_multiscale.get_knn_contexts(learning_data.mte_parameters["vc_like_data"].learning_settings_64_multiscale.sights[0])

        # Histogram matching data
        self.histogram_matching_data = learning_data.mte_parameters["vc_like_data"].histogram_matching_data
  
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
            # cv2.imshow("Sight", sight_image) # Debug

            for j, roi in enumerate(sight.roi):
                image = Image()
                image.sight_position = Point2D()
                image.sight_position.x = pt_tl.x
                image.sight_position.y = pt_tl.y
                image.image_class = image_class

                image_filter = ImageFilterType(roi.image_filter_type)

                detector = LinesDetector(sight_image, image_filter)
                mask = detector.detect()
                # cv2.imshow("Sight mask", mask) # Debug

                x = int(roi.x)
                y = int(roi.y)
                width = int(roi.width)
                height = int(roi.height)

                roi_mask = mask[y:y+height, x:x+width]
                # cv2.imshow("ROI"+str(j), roi_mask) # Debug

                # Feature extraction
                feature_vector = roi.feature_type
                vector = BoxLearner.extract_pixels_features(roi_mask, ROIFeatureType(feature_vector))

                feature = ROIFeature()
                feature.feature_type = ROIFeatureType(feature_vector)
                feature.feature_vector = vector[0].tolist()

                image.features.append(feature)

                roi.images.append(image)
        
        # cv2.waitKey(0) # Debug

    def mean_best(self, matches, best_match):
        min_means = 9999999
        ite_x = self.learning_settings_85.sights[0].search_box.iteration.x
        ite_y = self.learning_settings_85.sights[0].search_box.iteration.y
        for i in range(1, ite_x-1):
            for j in range(1, ite_y-1):
                if matches[i*ite_y + j].success:
                    temp = (matches[(i-1)*ite_y + j-1].max_distance +\
                            matches[(i-1)*ite_y + j].max_distance +\
                            matches[(i-1)*ite_y + j+1].max_distance +\
                            matches[(i)*ite_y + j-1].max_distance +\
                            matches[(i)*ite_y + j].max_distance +\
                            matches[(i)*ite_y + j+1].max_distance +\
                            matches[(i+1)*ite_y + j-1].max_distance +\
                            matches[(i+1)*ite_y + j].max_distance +\
                            matches[(i+1)*ite_y + j+1].max_distance) / 9
                    if temp < min_means:
                        min_means = temp
                        best_match = matches[(i)*ite_y + j]
        # for m in matches:
        #     print(m.anchor.x)
            # if m.anchor.x == 1
        return best_match

    def find_target(self, input_image, learning_data, testing_mode=False):
        not_centered = False
        capture = False
        response_type = MTEResponse.ORANGE
        translation = (0, 0)

        fps = FPS().start() # Debug

        image = cv2.resize(input_image, (self.image_width, self.image_height))
        if not testing_mode and not self.disable_histogram_matching:
            image = self.match_histograms(image, self.histogram_matching_data)

            # if self.debug_mode:
            #     cv2.imshow("Histogram matching correction", cv2.resize(image, (self.image_width*2, self.image_height*2)))
            #     cv2.waitKey(1)

        begin_timeout = time.time()

        number_of_green_around = 0
        step_done = False

        # Boîte verte
        if self.mode <= 0 or (self.nb_frames >= 9 and not self.one_shot_mode) or testing_mode:
            if self.mode == 0:
                prev_mode = 0

            self.validation_count = 0
            # Scan global
            best_match, matches, all_matches, green_matches, light_green_matches, orange_matches, to_display = self.box_learners_85_singlescale[self.scale].scan(image, scan_opti=False, output_matches=True)
            best_match = self.mean_best(all_matches, best_match)
            if len(matches) == 0:
                response_type = MTEResponse.TARGET_LOST
            if best_match.success:
                # Calcul du scale
                pt_tl = Point2D()
                pt_tl.x = best_match.anchor.x - self.box_learner_85_multiscale.sight.anchor.x
                pt_tl.y = best_match.anchor.y - self.box_learner_85_multiscale.sight.anchor.y

                pt_br = Point2D()
                pt_br.x = pt_tl.x + self.box_learner_85_multiscale.sight.width
                pt_br.y = pt_tl.y + self.box_learner_85_multiscale.sight.height

                self.box_learner_85_multiscale.input_image = image
                multiscale_match = self.box_learner_85_multiscale.find_target(pt_tl, pt_br, skip_tolerance=True)
                self.scale = multiscale_match.predicted_class

                translation = (int((pt_tl.x - (self.image_width/2 - self.box_learner_85_multiscale.sight.width/2))*(self.scale/100)), \
                    int((pt_tl.y - (self.image_height/2 - self.box_learner_85_multiscale.sight.height/2))*(self.scale/100)))

                # Change mode if there are 5 green points around and the target is roughly centered
                if not testing_mode:
                    green_count = 0
                    x1 = best_match.anchor.x
                    y1 = best_match.anchor.y
                    for m in matches:
                        x2 = m.anchor.x
                        y2 = m.anchor.y
                        if m.success:
                            green_count += 1
                        if (math.sqrt(pow(x2-x1, 2) + pow(y2-y1, 2)) < 15) and m.success:
                            number_of_green_around += 1
                    ratio_width_to_height = ((image.shape[1]/2)*(1/10)) / (image.shape[0]/2)
                    if (number_of_green_around >= 3) and (self.one_shot_mode or
                        math.isclose(best_match.anchor.x, image.shape[1]/2, rel_tol=float(10)/100) and\
                        math.isclose(best_match.anchor.y, image.shape[0]/2, rel_tol=ratio_width_to_height)):

                        # Change mode only if it is not the 10th frame check
                        if self.mode == 0:
                            self.mode = 1
                        else:
                            best_match = self.last_match
                            response_type = self.last_response_type
                            translation = self.last_translation
                    else:
                        if (time.time() - begin_timeout) > TIMEOUT_LIMIT_SEC:
                            response_type = MTEResponse.TARGET_LOST
                        elif (time.time() - begin_timeout) > (TIMEOUT_LIMIT_SEC / 2):
                            response_type = MTEResponse.RED
                        self.mode = 0

            if best_match.success:
                self.last_match = best_match

            step_done = True

        # Boîte orange
        if self.mode == 1 and (not step_done or self.one_shot_mode):
            prev_mode = 1
            response_type = MTEResponse.GREEN
            # Scan optimisé (step=3)
            best_match, matches, green_matches, light_green_matches, orange_matches, to_display = self.box_learners_64_singlescale[self.scale].optimised_scan_sequenced(image, best_match=self.last_match, output_matches=True)

            if best_match.success:
                # Calcul du scale
                pt_tl = Point2D()
                pt_tl.x = best_match.anchor.x - self.box_learner_64_multiscale.sight.anchor.x
                pt_tl.y = best_match.anchor.y - self.box_learner_64_multiscale.sight.anchor.y

                pt_br = Point2D()
                pt_br.x = pt_tl.x + self.box_learner_64_multiscale.sight.width
                pt_br.y = pt_tl.y + self.box_learner_64_multiscale.sight.height

                self.box_learner_64_multiscale.input_image = image
                multiscale_match = self.box_learner_64_multiscale.find_target(pt_tl, pt_br, skip_tolerance=True)
                self.scale = multiscale_match.predicted_class

                translation = (int((pt_tl.x - (self.image_width/2 - self.box_learner_64_multiscale.sight.width/2))*(self.scale/100)), \
                    int((pt_tl.y - (self.image_height/2 - self.box_learner_64_multiscale.sight.height/2))*(self.scale/100)))

                # Change mode if there are 5 green points around and the target is roughly centered
                green_count = 0
                x1 = best_match.anchor.x
                y1 = best_match.anchor.y
                if not self.one_shot_mode and (not math.isclose(x1, image.shape[1]/2, rel_tol=1/1) or\
                    not math.isclose(y1, image.shape[0]/2, rel_tol=1/1)):
                    not_centered = True
                    response_type = MTEResponse.ORANGE
                    begin_timeout = time.time()
                    self.mode = 0
                else: 
                    for m in matches:
                        x2 = m.anchor.x
                        y2 = m.anchor.y
                        if m.success:
                            green_count += 1
                        if (math.sqrt(pow(x2-x1, 2) + pow(y2-y1, 2)) < 10) and m.success:
                            number_of_green_around += 1
                    if number_of_green_around >= 4:
                        self.mode = 2
            else:
                response_type = MTEResponse.ORANGE
                begin_timeout = time.time()
                self.mode = 0

            if best_match.success:
                self.last_match = best_match

            step_done = True

        # Boîte rose
        if self.mode in (2, 3) and (not step_done or self.one_shot_mode):
            prev_mode = self.mode
            response_type = MTEResponse.GREEN

            # Scan optimisé (step=1)
            best_match, matches, green_matches, light_green_matches, orange_matches, to_display = self.box_learners_64_singlescale[self.scale].optimised_scan_sequenced(image, best_match=self.last_match, pixel_scan=True, output_matches=True)

            if best_match.success:
                # Calcul du scale
                pt_tl = Point2D()
                pt_tl.x = best_match.anchor.x - self.box_learner_64_multiscale.sight.anchor.x
                pt_tl.y = best_match.anchor.y - self.box_learner_64_multiscale.sight.anchor.y

                pt_br = Point2D()
                pt_br.x = pt_tl.x + self.box_learner_64_multiscale.sight.width
                pt_br.y = pt_tl.y + self.box_learner_64_multiscale.sight.height

                self.box_learner_64_multiscale.input_image = image
                multiscale_match = self.box_learner_64_multiscale.find_target(pt_tl, pt_br, skip_tolerance=True)
                self.scale = multiscale_match.predicted_class
                
                translation = (int((pt_tl.x - (self.image_width/2 - self.box_learner_64_multiscale.sight.width/2))*(self.scale/100)), \
                    int((pt_tl.y - (self.image_height/2 - self.box_learner_64_multiscale.sight.height/2))*(self.scale/100)))

                green_count = 0
                x1 = best_match.anchor.x
                y1 = best_match.anchor.y
                if not self.one_shot_mode and (not math.isclose(x1, image.shape[1]/2, rel_tol=1/1) or\
                    not math.isclose(y1, image.shape[0]/2, rel_tol=1/1)):
                    begin_timeout = time.time()
                    response_type = MTEResponse.ORANGE
                    not_centered = True
                    self.mode = 0
                else:
                    for m in matches:
                        x2 = m.anchor.x
                        y2 = m.anchor.y
                        if m.success:
                            green_count += 1
                        if (math.sqrt(pow(x2-x1, 2) + pow(y2-y1, 2)) < 10) and m.success:
                            number_of_green_around += 1
                # Change mode if there are 6 green points around and the target is roughly centered
                    if number_of_green_around >= 6:
                        self.mode = 3
                    else:
                        self.mode = 2
            else:
                self.mode = 1
            
            if prev_mode == 3 and response_type == MTEResponse.GREEN:
                prev_mode = 3

                self.validation_count += 1
                capture = (self.validation_count >= self.nb_following_captures)
                if capture:
                    response_type = MTEResponse.CAPTURE
        
            if best_match.success:
                self.last_match = best_match

            step_done = True
        
        self.last_response_type = response_type
        self.last_translation = translation

        if self.debug_mode and best_match.anchor is not None:
            x1 = best_match.anchor.x
            y1 = best_match.anchor.y
            # test = to_display.copy()
            for i in range (-2, 3):
                for j in range (-2, 3):
                    x = x1 + i
                    y = y1 + j
                    if abs(i) == abs(j):
                        to_display[y, x] = (255, 0, 0)

            # cv2.circle(to_display, (x1, y1), 2, (0, 255, 0), 2) # Debug
            # cv2.circle(test, (x1, y1), 2, (0, 0, 255), 2) # Debug

            # x1 = translation[0]
            # y1 = translation[1]
            # cv2.circle(to_display, (x1, y1), 2, (255, 0, 0), 2) # Debug
            # cv2.circle(test, (x1, y1), 2, (255, 0, 0), 2) # Debug
            # lower_right_corner = (int(x1+image.shape[1]), \
            #                                 int(y1+image.shape[0]))
            # to_display = cv2.rectangle(to_display, translation,\
            #                                     lower_right_corner, (255, 0, 0), thickness=1)
            # Scaled display
            # upper_left_conner = (int(x1-(image.shape[1])*(self.scale/100)), \
            #                                 int(y1-(image.shape[0])*(self.scale/100)))
            # lower_right_corner = (int(x1+(image.shape[1])*(self.scale/100)), \
            #                                 int(y1+(image.shape[0])*(self.scale/100)))
            # to_display = cv2.rectangle(to_display, upper_left_conner,\
            #                                     lower_right_corner, (255, 255, 255), thickness=1)
            cv2.imshow("Scan", cv2.resize(to_display, (self.image_width*2, self.image_height*2))) # Debug
            cv2.waitKey(1) # Debug
        # else:
        #     to_display = image.copy()
        #     test = to_display

        if not testing_mode:
            self.nb_frames += 1

        if self.nb_frames >= 10:
            self.nb_frames = 0

        fps.update() # Debug
        fps.stop() # Debug
        if best_match.success:
            print("Step = {}, X,Y = {},{}, Scale = {}, Dist = {}, Nb Success = {}, Nb green neighbours = {}, Green = {}, Lightgreen = {}, Orange = {}".\
                format(prev_mode, best_match.anchor.x, best_match.anchor.y, self.scale, best_match.max_distance, len(matches), number_of_green_around, green_matches, light_green_matches, orange_matches))
            lenght = len(matches)
            # writer.writerow({'Frame Id' : frame_id,
            #                     'Step' : prev_mode,
            #                     'X' : best_match.anchor.x,
            #                     'Y' : best_match.anchor.y,
            #                     'Scale' : self.scale,
            #                     'Dist' : int(best_match.max_distance),
            #                     'Nb Success' : lenght,
            #                     'Green' : green_matches,
            #                     'L Green' : light_green_matches,
            #                     'Orange' : orange_matches,
            #                     'Capture' : capture})
        else:
            print("Step = {}, Failure".\
                format(prev_mode))
            # writer.writerow({'Step' : prev_mode})

        # ite += 1
        # frame_id += 1
        # end_frame_computing = time.time()
        # if mode_skip == "fixe":
        #     for cpt in range(int((1/self.to_skip)-1)):
        #         cap.grab()
        # else:
        #     my_shift = end_frame_computing-begin_frame_computing
        #     self.to_skip = math.floor(my_shift*30)
        #     if fps.fps() < 30 and self.to_skip > 0:
        #         for cpt in range(self.to_skip):
        #             cap.grab()
        # print("Scale: {}".format(self.scale)) # Debug
        # print("FPS: {}".format(fps.fps())) # Debug
        # Display the resulting frame
        # cv2.imshow('Original image', image)

        # Press Q on keyboard to  exit
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        #TODO: remettre transformed ?
        # return best_match.success, (self.scale, self.scale), (0, 0), (best_match.anchor.x, best_match.anchor.y), transformed

        return best_match.success, response_type, (float(self.scale)/100, float(self.scale)/100), (0, 0), translation, input_image
    
    def generate_histogram_data(self, template):
        histogram_matching_data = []
        for channel in range(template.shape[-1]):
            tmpl_values, tmpl_counts = np.unique(template[..., channel].ravel(), return_counts=True)
            tmpl_quantiles = np.cumsum(tmpl_counts) / template[..., channel].size

            histogram_data = HistogramMatchingData()
            histogram_data.values = tmpl_values.tolist()
            histogram_data.counts = tmpl_counts.tolist()
            histogram_data.quantiles = tmpl_quantiles.tolist()

            histogram_matching_data.append(histogram_data)
        
        return histogram_matching_data
        
    def _match_cumulative_cdf(self, source, histogram_data:HistogramMatchingData):
        src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                            return_inverse=True,
                                                            return_counts=True)

        # calculate normalized quantiles for each array
        src_quantiles = np.cumsum(src_counts) / source.size

        tmpl_quantiles = np.asarray(histogram_data.quantiles, dtype=np.float64)
        tmpl_values = np.asarray(histogram_data.values, dtype=np.uint8)

        interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
        return interp_a_values[src_unique_indices].reshape(source.shape)
        
    def match_histograms(self, image, histogram_matching_data):
        matched = np.empty(image.shape, dtype=image.dtype)
        for channel in range(image.shape[-1]):
            matched_channel = self._match_cumulative_cdf(image[..., channel], histogram_matching_data[channel])
            matched[..., channel] = matched_channel

        return matched

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
