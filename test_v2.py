#!/usr/bin/env python3

import os
import sys
import operator
import glob
import math
import json
from datetime import datetime
import csv
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pykson import Pykson

from ML.Domain.LearningKnowledge import LearningKnowledge
from ML.Domain.Image import Image
from ML.Domain.ImageClass import ImageClass
from ML.Domain.ROIFeatureType import ROIFeatureType
from ML.Domain.ROIFeature import ROIFeature
from ML.Domain.ImageFilterType import ImageFilterType
from ML.Domain.Point2D import Point2D
from ML.Domain.Sight import Sight
from ML.Domain.RecognitionFlag import RecognitionFlag

from ML.BoxLearner import BoxLearner
from ML.LinesDetector import LinesDetector

from imutils.video import FPS

LEARNING_SETTINGS_85 = "learning_settings_85.json"
LEARNING_SETTINGS_64 = "learning_settings_64.json"

REFERENCE_IMAGE_PATH = "videos/short_magnetron_ref.png"
VIDEO_PATH = "videos/short_magnetron.mp4"

class Test():
    def __init__(self):
        self.reference_image = cv2.resize(cv2.imread(REFERENCE_IMAGE_PATH), (176, 97))

        self.learn()

        self.last_match = None
        self.mode = 0
        self.scale = 100
        self.nb_frames = 0

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
        
    def learn(self):
        self.dataset = self.generate_dataset(self.reference_image)

        learning_settings_85 = self.load_ml_settings(LEARNING_SETTINGS_85)
        learning_settings_64 = self.load_ml_settings(LEARNING_SETTINGS_64)

        # 85x48 learnings
        self.box_learners_85_singlescale = {}
        learning_settings_85_multiscale = deepcopy(learning_settings_85)

        # 64x64 learnings
        self.box_learners_64_singlescale = {}
        learning_settings_64_multiscale = deepcopy(learning_settings_64)

        for i, data in enumerate(self.dataset):
            attr, image = data[:]

            scale = int(attr["scale"]*100)

            class_id = scale
            class_name = "scale: {}".format(scale)

            # Create every 85x48 single-scale box learners
            learning_settings_singlescale_85 = deepcopy(learning_settings_85)
            self.learn_image(learning_settings_singlescale_85, class_id, class_name, image)

            box_learner_singlescale_85 = BoxLearner(learning_settings_singlescale_85.sights, 0)
            box_learner_singlescale_85.get_knn_contexts(learning_settings_singlescale_85.sights[0])

            self.box_learners_85_singlescale[scale] = box_learner_singlescale_85

            # Add learning to the 85x48 multi-scale learning
            self.learn_image(learning_settings_85_multiscale, class_id, class_name, image)

            # Create every 64x64 single-scale box learners
            learning_settings_singlescale_64 = deepcopy(learning_settings_64)
            self.learn_image(learning_settings_singlescale_64, class_id, class_name, image)
            
            box_learner_singlescale_64 = BoxLearner(learning_settings_singlescale_64.sights, 0)
            box_learner_singlescale_64.get_knn_contexts(learning_settings_singlescale_64.sights[0])

            self.box_learners_64_singlescale[scale] = box_learner_singlescale_64

            # Add learning to the 64x64 multi-scale learning
            self.learn_image(learning_settings_64_multiscale, class_id, class_name, image)
        
        # Create 85x48 multi-scale box learner
        self.box_learner_85_multiscale = BoxLearner(learning_settings_85_multiscale.sights, 0)
        self.box_learner_85_multiscale.get_knn_contexts(learning_settings_85_multiscale.sights[0])

        # Create 64x64 multi-scale box learner
        self.box_learner_64_multiscale = BoxLearner(learning_settings_64_multiscale.sights, 0)
        self.box_learner_64_multiscale.get_knn_contexts(learning_settings_64_multiscale.sights[0])
        
    
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

    def main(self):
        cap = cv2.VideoCapture(VIDEO_PATH)

        # Check if camera opened successfully
        if not cap.isOpened(): 
            print("Error opening video stream or file")

        # Read until video is completed
        while cap.isOpened():
            # Capture frame-by-frame
            success, image = cap.read()

            if success:
                fps = FPS().start() # Debug

                image = cv2.resize(image, (176, 97))

                # Boîte verte
                if self.mode <= 0 or self.nb_frames >= 10:
                    # Scan global
                    best_match, matches = self.box_learners_85_singlescale[self.scale].scan(image, scan_opti=False, output_matches=True)

                    if best_match.success:
                        # Calcul du scale
                        pt_tl = Point2D()
                        pt_tl.x = best_match.anchor.x - self.box_learner_85_multiscale.sight.anchor.x
                        pt_tl.y = best_match.anchor.y - self.box_learner_85_multiscale.sight.anchor.y

                        pt_br = Point2D()
                        pt_br.x = pt_tl.x + self.box_learner_85_multiscale.sight.width
                        pt_br.y = pt_tl.y + self.box_learner_85_multiscale.sight.height

                        self.box_learner_85_multiscale.input_image = image
                        multiscale_match = self.box_learner_85_multiscale.find_target(pt_tl, pt_br)

                        self.scale = multiscale_match.predicted_class

                        #TODO: changement de mode si 5 points verts proches (regarder leur .anchor, tous les matches sont dans la variable matches) 
                        #TODO: et cible à peu près au centre de l'image
                        self.mode = 1

                # Boîte orange
                elif self.mode == 1:
                    # Scan optimisé (step=3)
                    best_match, matches = self.box_learners_64_singlescale[self.scale].optimised_scan_sequenced(image, best_match=self.last_match, output_matches=True)

                    if best_match.success:
                        # Calcul du scale
                        pt_tl = Point2D()
                        pt_tl.x = best_match.anchor.x - self.box_learner_64_multiscale.sight.anchor.x
                        pt_tl.y = best_match.anchor.y - self.box_learner_64_multiscale.sight.anchor.y

                        pt_br = Point2D()
                        pt_br.x = pt_tl.x + self.box_learner_64_multiscale.sight.width
                        pt_br.y = pt_tl.y + self.box_learner_64_multiscale.sight.height

                        self.box_learner_64_multiscale.input_image = image
                        multiscale_match = self.box_learner_64_multiscale.find_target(pt_tl, pt_br)

                        self.scale = multiscale_match.predicted_class

                        #TODO: changement de mode si 5 points verts proches (regarder leur .anchor, tous les matches sont dans la variable matches) 
                        #TODO: et cible à peu près au centre de l'image
                        self.mode = 2
                    else:
                        #TODO: définir la condition pour la redescente de mode (oubli dans le diagramme d'activité)
                        self.mode = 0

                # Boîte rose
                else:
                    # Scan optimisé (step=1)
                    best_match, matches = self.box_learners_64_singlescale[self.scale].optimised_scan_sequenced(image, best_match=self.last_match, pixel_scan=True, output_matches=True)

                    if best_match.success:
                        # Calcul du scale
                        pt_tl = Point2D()
                        pt_tl.x = best_match.anchor.x - self.box_learner_64_multiscale.sight.anchor.x
                        pt_tl.y = best_match.anchor.y - self.box_learner_64_multiscale.sight.anchor.y

                        pt_br = Point2D()
                        pt_br.x = pt_tl.x + self.box_learner_64_multiscale.sight.width
                        pt_br.y = pt_tl.y + self.box_learner_64_multiscale.sight.height

                        self.box_learner_64_multiscale.input_image = image
                        multiscale_match = self.box_learner_64_multiscale.find_target(pt_tl, pt_br)
                        self.scale = multiscale_match.predicted_class

                        #TODO: passage à SIFT
                    else:
                        #TODO: définir la condition pour la redescente de mode (oubli dans le diagramme d'activité)
                        self.mode = 1
                
                #TODO: boîte jaune (SIFT)
                """ elif mode 3
                    si self.sclale == 100
                        calcul des descripteurs des 2 images (avec videos/ref)""" 

                self.last_match = best_match
                
                if self.nb_frames >= 10:
                    self.nb_frames = 0
                else:
                    self.nb_frames += 1

                fps.update() # Debug
                fps.stop() # Debug

                print("Scale: {}".format(self.scale)) # Debug
                print("FPS: {}".format(fps.fps())) # Debug

                # Display the resulting frame
                # cv2.imshow('Original image', image)

                # Press Q on keyboard to  exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Break the loop
            else: 
                break

        # When everything done, release the video capture object
        cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()




if __name__ == "__main__":
    app = Test()
    app.main()