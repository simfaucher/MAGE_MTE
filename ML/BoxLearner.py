#!/usr/bin/env python3

"""
    Machine learning for feature recognition.

    Project Nose Landing Gear Video Measurement for ATR
    Created on Mon Oct 21 2019 by Frank Ben Zaquin, Fabien Monniot
    Copyright (c) 2019 Altran Technologies
"""

import math
import time
import json
import glob
import statistics

# from numba import jit

import cv2
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from ML.Domain.LearningKnowledge import LearningKnowledge
from ML.Domain.Image import Image
from ML.Domain.ROIFeatureType import ROIFeatureType
from ML.Domain.ROIFeature import ROIFeature
from ML.Domain.ImageFilterType import ImageFilterType
from ML.Domain.Point2D import Point2D
from ML.Domain.ROIFeatureType import ROIFeatureType
from ML.Domain.Sight import Sight
from ML.Domain.Step import Step
from ML.Domain.RecognitionFlag import RecognitionFlag
from ML.Domain.LearnerMatch import LearnerMatch

from ML.LinesDetector import LinesDetector

class BoxLearner():
    def __init__(self, sights: list, uncertainty: float = 0):
        self.sight = None
        self.sights = sights
        self.input_image = None

        self.knn_contexts = []
        self.learning_classes = []
        self.uncertainty = uncertainty

    def get_knn_contexts(self, sight: Sight):
        self.sight = sight
        self.knn_contexts = []
        self.learning_classes = []

        # One context for each ROI
        for roi in sight.roi:
            # Get the features
            features = []
            classes = []
            for image in roi.images:
                to_delete = []

                # If there are features
                if image.features:
                    for feature in image.features:
                        if ROIFeatureType(feature.feature_type) == ROIFeatureType(roi.feature_type):
                            # Uncertainty
                            keep_feature = True
                            if self.uncertainty != 0:
                                for saved_feature in features:
                                    distance = np.linalg.norm(np.array(feature.feature_vector) - np.array(saved_feature))

                                    if distance < self.uncertainty:
                                        keep_feature = False
                                        break

                            if keep_feature:
                                features.append(feature.feature_vector)
                                classes.append(image.image_class.id)

                            # Will be deleted from sight if below uncertainty
                            else:
                                to_delete.append(feature)

                        # Will be deleted from sight if it is not the current feature type
                        else:
                            to_delete.append(feature)

                # Remove the useless features
                for feature in to_delete:
                    image.features.remove(feature)

            features = np.array(features).astype(np.float32)
            classes = np.array(classes).astype(np.float32).ravel()

            # Create the kNN context associated to this ROI
            knn_context = KNeighborsClassifier(n_neighbors=len(features), metric="manhattan")
            knn_context.fit(features, classes)
            # print(knn_context.get_params())

            self.knn_contexts.append(knn_context)
            self.learning_classes.append(classes)

    # @jit
    def scan(self, input_image, start_point: Point2D = None, scan_opti=True):
        self.input_image = input_image

        matches = []

        search_box = self.sight.search_box

        if start_point is not None:
            search_box.anchor = Point2D()
            search_box.anchor.x = start_point.x - self.sight.anchor.x
            search_box.anchor.y = start_point.y - self.sight.anchor.y
        elif search_box.anchor is None:
            search_box.anchor = Point2D()
            search_box.anchor.x = 0
            search_box.anchor.y = 0

        for i in range(0, search_box.iteration.x):
            for j in range(0, search_box.iteration.y):
                point_tl = Point2D()
                point_tl.x = search_box.anchor.x + i * search_box.step.x
                point_tl.y = search_box.anchor.y + j * search_box.step.y

                point_br = Point2D()
                point_br.x = point_tl.x + self.sight.width - 1
                point_br.y = point_tl.y + self.sight.height - 1

                # Debug
                # if i == 0 and j == 0:
                #     print("ROS normal scan first pt tl + anchor: x:{}, y:{}".format(point_tl.x + self.sight.anchor.x, point_tl.y + self.sight.anchor.y))
                # if i == search_box.iteration.x - 1 and j == search_box.iteration.y - 1:
                #     print("ROS normal scan last pt tl + anchor: x:{}, y:{}".format(point_br.x + self.sight.anchor.x, point_br.y + self.sight.anchor.y))

                if not self.is_position_valid(input_image, point_tl, point_br):
                    continue

                match = self.find_target(point_tl, point_br, skip_tolerance=True)

                # debug_image = input_image[point_tl.y: point_br.y, point_tl.x: point_br.x]
                # cv2.imshow("Scaning", debug_image)
                # print("Scaning distance: {}".format(match.max_distance))
                # cv2.waitKey(0)

                if match.success:
                    matches.append(match)

        # Get the best match
        best_match = self.get_best_match(matches)

        # print("===============") # Debug
        # print("sum_distances"+ str(best_match.sum_distances)) # Debug
        # cv2.waitKey(0) # Debug

        # Debug
        # sight_image = input_image[best_match.anchor.y - self.sight.anchor.y \
        #     :best_match.anchor.y - self.sight.anchor.y + self.sight.height, \
        #         best_match.anchor.x - self.sight.anchor.x\
        #             :best_match.anchor.x - self.sight.anchor.x + self.sight.width]
        # for k, roi in enumerate(self.sight.roi):
        #     image_filter = ImageFilterType(roi.image_filter_type)

        #     detector = LinesDetector(sight_image, image_filter)
        #     mask = detector.detect()

        #     x = int(roi.x)
        #     y = int(roi.y)
        #     width = int(roi.width)
        #     height = int(roi.height)

        #     roi_mask = mask[y:y+height, x:x+width]
        #     cv2.imshow("ROI"+str(k), roi_mask)
        #     cv2.imshow("ROI_raw_pixels16 "+str(k), cv2.resize(roi_mask, (16, 16)))

        # cv2.imshow("Sight image", cv2.cvtColor(sight_image, cv2.COLOR_BGR2GRAY))
        # print("BEST_MATCH : x:{}, y:{}".format(best_match.anchor.x, best_match.anchor.y))

        # cv2.waitKey(0) # Debug
        # cv2.destroyAllWindows()
        # End debug

        # Optimised scan to find pixel sensitive best position
        if best_match.success and scan_opti:
            best_match = self.optimised_scan(input_image, best_match)
        else:
            # Find the recognition flag
            best_match.power_of_recognition = self.get_recognition_flag(best_match)

        best_match.reduced = True

        return best_match

    # @jit
    def optimised_scan(self, input_image, best_match: LearnerMatch = None, anchor_point: Point2D = None, pixel_scan=False):
        self.input_image = input_image

        step = Step()
        if pixel_scan:
            step.x = 3
            step.y = 3
        elif anchor_point:
            step.x = self.sight.search_box.step.x
            step.y = self.sight.search_box.step.y
        else:
            step.x = int(self.sight.search_box.step.x / 2)
            step.y = int(self.sight.search_box.step.y / 2)

        # Get the position from the best match of a normal scan
        if best_match is not None:
            anchor_point_tl = Point2D()
            anchor_point_tl.x = best_match.anchor.x - self.sight.anchor.x
            anchor_point_tl.y = best_match.anchor.y - self.sight.anchor.y
        # Get the position entered as input
        elif anchor_point is not None:
            # anchor_point_tl = anchor_point
            anchor_point_tl = Point2D()
            anchor_point_tl.x = anchor_point.x - self.sight.anchor.x
            anchor_point_tl.y = anchor_point.y - self.sight.anchor.y
        # Default mode
        else:
            anchor_point_tl = Point2D()
            anchor_point_tl.x = 0
            anchor_point_tl.y = 0

        hit_in = 0

        while step.x >= 1 and step.y >= 1:
            matches = []
            for i in range(-4, 5):
                for j in range(-4, 5):
                    point_tl = Point2D()
                    point_tl.x = anchor_point_tl.x + i * step.x
                    point_tl.y = anchor_point_tl.y + j * step.y

                    point_br = Point2D()
                    point_br.x = point_tl.x + self.sight.width
                    point_br.y = point_tl.y + self.sight.height

                    # Debug
                    # if i == -4 and j == -4:
                    #     print("ROS opti first pt tl + anchor : x:{}, y:{}".format(point_tl.x + self.sight.anchor.x, point_tl.y + self.sight.anchor.y))
                    # if i == 4 and j == 4:
                    #     print("ROS opti last pt tl + anchor : x:{}, y:{}".format(point_tl.x + self.sight.anchor.x, point_tl.y + self.sight.anchor.y))

                    if not self.is_position_valid(input_image, point_tl, point_br):
                        continue

                    match = self.find_target(point_tl, point_br)

                    # debug_image = input_image[point_tl.y: point_br.y, point_tl.x: point_br.x]
                    # cv2.imshow("Scaning", debug_image)
                    # print("Scaning distance: {}".format(match.max_distance))
                    # cv2.waitKey(0)

                    if match.success:
                        matches.append(match)

                        if step.x == 1 and step.y == 1:
                            hit_in += 1

            # More sensitive scan
            step.x = int(step.x / 2)
            step.y = int(step.y / 2)

            if step.y < 1 <= step.x:
                step.y = 1
            elif step.x < 1 <= step.y:
                step.x = 1

            # Get the best match
            best_match = self.get_best_match(matches)
            best_match.hit_in = hit_in
            if best_match.anchor is not None:
                anchor_point_tl.x = best_match.anchor.x - self.sight.anchor.x
                anchor_point_tl.y = best_match.anchor.y - self.sight.anchor.y

        # Find the recognition flag
        best_match.power_of_recognition = self.get_recognition_flag(best_match)

        # Debug
        # if best_match.success:
        #     sight_image = input_image[best_match.anchor.y - self.sight.anchor.y \
        #         :best_match.anchor.y - self.sight.anchor.y + self.sight.height, \
        #             best_match.anchor.x - self.sight.anchor.x\
        #                 :best_match.anchor.x - self.sight.anchor.x + self.sight.width]
        #     for k, roi in enumerate(self.sight.roi):
        #         image_filter = ImageFilterType(roi.image_filter_type)

        #         detector = LinesDetector(sight_image, image_filter)
        #         mask = detector.detect()

        #         x = int(roi.x)
        #         y = int(roi.y)
        #         width = int(roi.width)
        #         height = int(roi.height)

        #         roi_mask = mask[y:y+height, x:x+width]
        #         cv2.imshow("ROI"+str(k), roi_mask)
        #         cv2.imshow("ROI_raw_pixels16 "+str(k), cv2.resize(roi_mask, (16, 16)))

        #     cv2.imshow("Sight image", cv2.cvtColor(sight_image, cv2.COLOR_BGR2GRAY))
        #     print("BEST_MATCH : x:{}, y:{}".format(best_match.anchor.x, best_match.anchor.y))

        #     cv2.waitKey(0) # Debug
        #     cv2.destroyAllWindows()
        # End debug

        return best_match

    def find_target(self, point_tl: Point2D, point_br: Point2D, skip_tolerance=False):
        sight_image = self.input_image[point_tl.y: point_br.y, point_tl.x: point_br.x]
        # debug_image = sight_image.copy() # Debug

        match = LearnerMatch()
        match.sum_distances = 0
        match.success = False
        match.max_distance = 0

        # Compute the distance for each ROI
        success = 0
        for k, roi in enumerate(self.sight.roi):
            roi_x1 = roi.x
            roi_y1 = roi.y
            roi_x2 = roi_x1 + roi.width
            roi_y2 = roi_y1 + roi.height

            # cv2.rectangle(debug_image, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255)) # Debug

            image_filter = ImageFilterType(roi.image_filter_type)
            detector = LinesDetector(sight_image, image_filter)

            sight_mask = detector.detect()

            roi_mask = sight_mask[roi_y1: roi_y2, roi_x1: roi_x2]
            features = BoxLearner.extract_pixels_features(roi_mask, ROIFeatureType(roi.feature_type))
            if "Scale finder" in self.sight.name: 
                cv2.imshow("roi_mask"+str(k), roi_mask) # Debug

            distances, indices = self.knn_contexts[k].kneighbors(features)
            # neighbours_classes = self.knn_contexts[k].predict(features)

            distance = distances[0][0]

            succeeded = False

            if skip_tolerance or distance <= roi.tolerance:
                succeeded = True
                success += 1

            match.sum_distances += float(distance)

            if distance > match.max_distance:
                match.max_distance = float(distance)

            match.roi_distances.append(distance)

            # Ignore -1 category
            closest_class = int(self.learning_classes[k][indices[0][0]])
            if closest_class == -1:
                success = 0
                break

            # Use uncertainty to evaluate if the element is close to another class
            if succeeded:
                keep_class = True
                if len(indices[0]) > 1:
                    for i, indice in enumerate(indices[0][1:]):
                        if int(self.learning_classes[k][indice]) != closest_class:
                            # Check for uncertainty
                            if distances[0][i+1] - distance < self.uncertainty:
                                keep_class = False
                                match.power_of_recognition = RecognitionFlag.ORANGE
                            break

                if keep_class or len(indices[0]) == 1:
                    match.roi_classes.append(closest_class)

        # If every ROI succeeded
        if success == len(self.sight.roi):
            match.success = True
            # match.anchor = point_tl
            match.anchor = Point2D()
            match.anchor.x = point_tl.x + self.sight.anchor.x
            match.anchor.y = point_tl.y + self.sight.anchor.y
            # match.anchor.x = point_tl.x
            # match.anchor.y = point_tl.y

            # Get more represented class
            if match.roi_classes:
                classes_set = set(match.roi_classes) # list with unique elements
                match.predicted_class = max(classes_set, key=match.roi_classes.count)
            else:
                match.predicted_class = -1

        # print("max_dist"+str(match.max_distance)) # Debug

        # cv2.circle(debug_image, (self.sight.anchor.x, self.sight.anchor.y), 2, (0, 255, 255))
        # cv2.imshow("debug_image", debug_image) # Debug
        cv2.waitKey(1) # Debug

        return match

    def get_best_match(self, matches):
        found_matches = len(matches) > 0

        if found_matches:
            min_distance = math.inf
            for match in matches:
                if match.max_distance < min_distance:
                    best_match = match
                    min_distance = match.max_distance
        else:
            best_match = LearnerMatch()
            best_match.success = False

        return best_match

    def is_position_valid(self, input_image, point_tl: Point2D, point_br: Point2D):
        height, width = input_image.shape[:2]

        if point_tl.x < width - 1 and point_br.x < width \
            and point_tl.y < height - 1 and point_br.y < height:
            return True

        return False

    def get_recognition_flag(self, best_match: LearnerMatch):
        if best_match.success and best_match.predicted_class != -1:
            sum_tolerances = sum(roi.tolerance for roi in self.sight.roi)
            tolerance_margin = 100 - (best_match.sum_distances / sum_tolerances * 100)

            if tolerance_margin < 10 or best_match.hit_in is not None and best_match.hit_in < 6:
                return RecognitionFlag.ORANGE
            elif tolerance_margin < 25 or best_match.hit_in is not None and best_match.hit_in < 16:
                return RecognitionFlag.LIGHTGREEN

            return RecognitionFlag.GREEN

        return RecognitionFlag.RED

    @staticmethod
    def get_worse_flag(flag1: RecognitionFlag, flag2: RecognitionFlag):
        if flag1 == RecognitionFlag.RED or flag2 == RecognitionFlag.RED:
            return RecognitionFlag.RED
        
        elif flag1 == RecognitionFlag.ORANGE or flag2 == RecognitionFlag.ORANGE:
            return RecognitionFlag.ORANGE

        elif flag1 == RecognitionFlag.LIGHTGREEN or flag2 == RecognitionFlag.LIGHTGREEN:
            return RecognitionFlag.LIGHTGREEN

        return RecognitionFlag.GREEN

    @staticmethod
    def extract_pixels_features(image, features_type: ROIFeatureType):
        nb_layers = len(image.shape) if len(image.shape) == 3 else 1

        # Pixels profile
        if features_type == ROIFeatureType.RAW_PIXELS:
            resized = cv2.resize(image, (8, 8))
            return resized.reshape(1, 8*8*nb_layers).astype(np.float32)

        elif features_type == ROIFeatureType.RAW_PIXELS_16:
            resized = cv2.resize(image, (16, 16))
            return resized.reshape(1, 16*16*nb_layers).astype(np.float32)

        elif features_type == ROIFeatureType.RAW_PIXELS_64_4:
            resized = cv2.resize(image, (64, 4))
            return resized.reshape(1, 64*4*nb_layers).astype(np.float32)

        elif features_type == ROIFeatureType.RAW_PIXELS_4_64:
            resized = cv2.resize(image, (4, 64))
            return resized.reshape(1, 4*64*nb_layers).astype(np.float32)

        # Composite profiles
        elif features_type == ROIFeatureType.COMPOSITE_PROFILE \
            or features_type == ROIFeatureType.COMPOSITE_PROFILE_64:
            resized3 = cv2.resize(image, (32, 32))
            mean1 = np.mean(resized3, axis=0)
            mean2 = np.mean(resized3, axis=1)
            resized = np.concatenate((mean1.flatten(), mean2.flatten()))
            return resized.reshape(1, (32+32)*nb_layers).astype(np.float32)

        elif features_type == ROIFeatureType.COMPOSITE_PROFILE_128:
            resized3 = cv2.resize(image, (64, 64))
            mean1 = np.mean(resized3, axis=0)
            mean2 = np.mean(resized3, axis=1)
            resized = np.concatenate((mean1.flatten(), mean2.flatten()))
            return resized.reshape(1, (64+64)*nb_layers).astype(np.float32)

        # Color histograms
        elif features_type == ROIFeatureType.COLOR_HIST:
            if nb_layers == 3:
                hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                return hist.reshape(1, 8*8*8).astype(np.float32)

            hist = cv2.calcHist([image], [0], None, [64], [0, 256])

            return hist.reshape(1, 64).astype(np.float32)

        # HOG
        else:
            return BoxLearner.hog(image)

    @staticmethod
    def hog(img):
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 9 # Number of bins
        bins = np.int32(bin_n*ang/(2*np.pi))

        bin_cells = []
        mag_cells = []

        cellx = celly = 8

        for i in range(0, int(img.shape[0]/celly)):
            for j in range(0, int(img.shape[1]/cellx)):
                bin_cells.append(bins[i*celly : i*celly+celly, j*cellx : j*cellx+cellx])
                mag_cells.append(mag[i*celly : i*celly+celly, j*cellx : j*cellx+cellx])   

        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= np.linalg.norm(hist) + eps

        hist *= 100

        hist = hist.reshape(1, hist.shape[0]).astype(np.float32)

        return hist
