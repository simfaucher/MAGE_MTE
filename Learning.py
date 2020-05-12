#!/usr/bin/env python3

"""
    Learns sights on the learning dataset.

    Project Nose Landing Gear Video Measurement for ATR
    Created on Mon Oct 21 2019 by Frank Ben Zaquin, Fabien Monniot
    Copyright (c) 2019 Altran Technologies
"""

import os
import sys
import operator
import glob
import json
from datetime import datetime

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

INPUT_IMAGES_FOLDER = "images/"
INPUT_IMAGES_TYPES = ("*.bmp", "*.jpeg", "*.jpg", "*.png")
CONFIG_SIGHTS_FILENAME = "learning_settings.json"
OUTPUT_FILENAME = "temp_learnt_targets.json"

class Learning():
    """ Create a knowledge database from images """

    # region Initiatisation

    def __init__(self):
        # Read the sights config file
        self.load_input_data()

        # Load input images
        self.load_images()

        # Initialize sight position
        self._sight_point_tl = (0, 0)
        self.sight_size = (0, 0)
        self.current_image_name = ""
        self.current_image = []
        self.current_class = []
        self.current_sight = []

        # Drawing variables
        self.drawing = False
        self.offset_from_sight = (0, 0)

        # Initialize output data
        self.load_output_data()

    @property
    def sight_point_tl(self):
        return self._sight_point_tl

    @sight_point_tl.setter
    def sight_point_tl(self, value):
        if value[0] == 0 and value[1] == 0:
            self._sight_point_tl = value
            return

        height, width = self.current_image.shape[:2]
        point_br = tuple(map(operator.add, value, (self.sight_size[1], self.sight_size[0])))

        if value[0] < 0:
            self._sight_point_tl = 0, self._sight_point_tl[1]
        elif point_br[0] >= width:
            self._sight_point_tl = width - \
                self.sight_size[1] - 1, self._sight_point_tl[1]
        else:
            self._sight_point_tl = value[0], self._sight_point_tl[1]

        if value[1] < 0:
            self._sight_point_tl = self._sight_point_tl[0], 0
        elif point_br[1] >= height:
            self._sight_point_tl = self._sight_point_tl[0], height - \
                self.sight_size[0] - 1
        else:
            self._sight_point_tl = self._sight_point_tl[0], value[1]

    @property
    def sight_point_br(self):
        inv_sight_size = (self.sight_size[1], self.sight_size[0])
        return tuple(map(operator.add, self.sight_point_tl, inv_sight_size))

    def load_data(self, json_data):
        try:
            return Pykson.from_json(json_data, LearningKnowledge, accept_unknown=True)
        except TypeError as error:
            sys.exit("Type error in {} with the attribute \"{}\". Expected {} but had {}.".format(error.args[0], error.args[1], error.args[2], error.args[3]))

    def load_input_data(self):
        try:
            print("Reading the input file : {}".format(CONFIG_SIGHTS_FILENAME))
            with open(CONFIG_SIGHTS_FILENAME) as json_file:
                json_data = json.load(json_file)
        except IOError as error:
            sys.exit("The file doesn't exist.")

        self.input_data = self.load_data(json_data)

    def load_output_data(self):
        try:
            with open(OUTPUT_FILENAME) as json_file:
                json_data = json.load(json_file)
            
            self.output_data = self.load_data(json_data)

            print("Read the previous output file : {}. \
                The sights position will be used if corresponding."\
                    .format(OUTPUT_FILENAME))
        except IOError as error:
            self.output_data = None
            print("No previous output file found, {} will be created.".format(OUTPUT_FILENAME))

    def load_images(self):
        self.input_images = []

        # Look for the dataset for each sight
        for i, sight in enumerate(self.input_data.sights):
            self.input_images.append([])

            sight_dir = os.path.join(INPUT_IMAGES_FOLDER, sight.name)
            if not os.path.exists(sight_dir):
                sys.exit("The path {} does not exist. Create the directory and fill it with images.".format(sight_dir))
            sight_classes = [os.path.join(sight_dir, o) for o in os.listdir(sight_dir) \
                if os.path.isdir(os.path.join(sight_dir, o))]

            # Each class of each sight
            for sight_class in sight_classes:
                class_basename = os.path.basename(sight_class)
                class_split = class_basename.split('-')

                if class_split[0]:
                    class_id = int(class_split[0])
                    class_name = class_split[1:]
                else:
                    class_id = -1
                    class_name = class_split[2:]
                class_name = '-'.join(class_name)


                image_class = ImageClass()
                image_class.id = class_id
                image_class.name = class_name
                class_dict = {
                    "image_class": image_class,
                    "images": []
                }

                self.input_images[i].append(class_dict)

                # Keep only the images
                for images_type in INPUT_IMAGES_TYPES:
                    dir_path = os.path.join(sight_class, images_type)
                    images_path = glob.glob(dir_path)

                    for image_path in images_path:

                        class_dict["images"].append(os.path.basename(image_path))

                class_dict["images"].sort()

    # endregion

    # region GUI

    def init_display(self):
        cv2.namedWindow("Learning")
        cv2.setMouseCallback("Learning", self.move_sight)

    def display(self):
        while True:
            display_image = self.current_image.copy()

            self.draw_sight(display_image, self.sight_point_tl)

            cv2.imshow("Learning", display_image)

            # Display the legend
            self.display_legend()

            key = cv2.waitKey(1)

            # Next image on spacebar push or exit program
            if key == 32:  # Spacebar
                return 0
            elif key == 27:  # escape
                return -1

            # Move sight with keyboard
            elif key == ord("q"):  # Left arrow
                self.sight_point_tl = tuple(
                    map(operator.add, self.sight_point_tl, (-1, 0)))
            elif key == ord("z"):  # Up arrow
                self.sight_point_tl = tuple(
                    map(operator.add, self.sight_point_tl, (0, -1)))
            elif key == ord("d"):  # Right arrow
                self.sight_point_tl = tuple(
                    map(operator.add, self.sight_point_tl, (1, 0)))
            elif key == ord("s"):  # Down arrow
                self.sight_point_tl = tuple(
                    map(operator.add, self.sight_point_tl, (0, 1)))

    def draw_sight(self, image, sight_point_tl):
        sight_point_br = (sight_point_tl[0] + self.sight_size[1] - 1, \
            sight_point_tl[1] + self.sight_size[0] - 1)

        # Draw sight
        cv2.rectangle(image, sight_point_tl, \
                        sight_point_br, (255, 0, 0), 1)

        # Draw ROIs
        for roi in self.current_sight.roi:
            point_tl = tuple( \
                map(operator.add, sight_point_tl, (int(roi.x), int(roi.y))))
            point_br = tuple(map(operator.add, point_tl, \
                                    (int(roi.width) - 1, int(roi.height) - 1)))

            cv2.rectangle(image, point_tl, point_br, (0, 0, 255), 1)

        # Draw anchor
        anchor_relative = (int(self.current_sight.anchor.x), int(
            self.current_sight.anchor.y))
        anchor = tuple(
            map(operator.add, sight_point_tl, anchor_relative))
        cv2.circle(image, (anchor), 3, (0, 255, 255))

    def display_legend(self):
        margin = 20
        line_height = 15
        w = self.sight_size[1] + margin*2
        w = w if self.sight_size[1] + margin*2 >= 200 else 200
        h = self.sight_size[0] + margin*3 + line_height*(len(self.current_sight.roi)+1)
        legend_image = np.ones((h, w, 3), dtype=np.uint8)*220

        self.draw_sight(legend_image, (margin, margin))

        # Write name of the sight
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Sight: {}".format(self.current_sight.name)
        cv2.putText(legend_image, text, \
            (margin, self.sight_size[0] + margin*2), \
                font, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

        # Write name of the sight class
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Class: {}-{}".format(self.current_class.id, self.current_class.name)
        cv2.putText(legend_image, text, \
            (margin, self.sight_size[0] + margin*2 + line_height), \
                font, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

        # Write anchor legend
        for i, roi in enumerate(self.current_sight.roi):
            text = "Anchor"
            cv2.putText(legend_image, text, \
                (margin, self.sight_size[0] + margin*2 + line_height * 2), \
                    font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        # Write name and number of the ROI
        for i, roi in enumerate(self.current_sight.roi):
            # Write name
            text = "ROI {}: {}".format(i, roi.name)
            cv2.putText(legend_image, text, \
                (margin, self.sight_size[0] + margin*2 + (i+3)*line_height), \
                    font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

            # Write id
            point_tl = tuple(
                map(operator.add, (margin, margin), (int(roi.x), int(roi.y))))
            point_tl = tuple(
                map(operator.add, point_tl, (int(int(roi.width)/2), int(int(roi.height)/2))))
            point_tl = tuple(
                map(operator.add, point_tl, (-3, 5)))
            text = str(i)
            cv2.putText(legend_image, text, point_tl, font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow("Legend", legend_image)

    def move_sight(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.drawing:
            # If the sight is selected
            if x > self.sight_point_tl[0] and x < self.sight_point_br[0] \
                    and y > self.sight_point_tl[1] and y < self.sight_point_br[1]:

                self.drawing = True
                self.offset_from_sight = tuple(
                    map(operator.sub, (x, y), self.sight_point_tl))

        if event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False

        if event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.sight_point_tl = tuple(
                map(operator.sub, (x, y), self.offset_from_sight))

    # endregion

    # region Learning

    def learn(self):
        quit_program = False
        for i, sight in enumerate(self.input_data.sights):
            if quit_program:
                break

            self.current_sight = sight

            for j, class_dict in enumerate(self.input_images[i]):
                if quit_program:
                    break

                self.current_class = class_dict["image_class"]

                for image_name in class_dict["images"]:
                    # Initialize image and sight
                    image_path = os.path.join(INPUT_IMAGES_FOLDER, self.current_sight.name)
                    image_path = os.path.join(image_path, str(self.current_class.id) + "-" + self.current_class.name, image_name)
                    print("Image {}".format(image_path))
                    self.current_image_name = image_path
                    self.current_image = cv2.imread(image_path)
                    height, width = self.current_image.shape[:2]

                    self.sight_size = (int(sight.height), int(sight.width))

                    # Get the previous output data
                    self.sight_point_tl = self.get_sight_image_position()

                    # Prepare display
                    self.init_display()
                    if self.display() == 0:
                        self.save_image_sight()
                    else:
                        quit_program = True
                        break

        # Write output JSON file
        self.input_data.image_folder = INPUT_IMAGES_FOLDER
        self.input_data.generation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        json_str = Pykson.to_json(self.input_data)
        with open(OUTPUT_FILENAME, 'w') as outfile:
            outfile.write(json_str)

        cv2.destroyAllWindows()

    def get_sight_image_position(self):
        # Get the existing data of the sight
        position = (0, 0)

        if self.output_data:
            output_data_sight = [x for x in self.output_data.sights \
                        if x.name == self.current_sight.name]

            if output_data_sight:
                # Get the existing data of the image
                output_data_image = [x for x in output_data_sight[0].roi[0].images \
                    if x.path == self.current_image_name]

                if output_data_image:
                    position = (output_data_image[0].sight_position.x, output_data_image[0].sight_position.y)

        return position

    def save_image_sight(self):
        sight_image = self.current_image[self.sight_point_tl[1] \
            :self.sight_point_br[1], self.sight_point_tl[0]:self.sight_point_br[0]]

        for i, roi in enumerate(self.current_sight.roi):
            image = Image()
            image.path = self.current_image_name
            image.sight_position = Point2D()
            image.sight_position.x = self.sight_point_tl[0]
            image.sight_position.y = self.sight_point_tl[1]
            image.image_class = self.current_class

            image_filter = ImageFilterType(roi.image_filter_type)

            detector = LinesDetector(sight_image, image_filter)
            mask = detector.detect()
            cv2.imshow("Sight", mask)

            x = int(roi.x)
            y = int(roi.y)
            width = int(roi.width)
            height = int(roi.height)

            roi_mask = mask[y:y+height, x:x+width]
            cv2.imshow("ROI"+str(i), roi_mask)

            for feature_vector in ROIFeatureType:
                vector = BoxLearner.extract_pixels_features(roi_mask, ROIFeatureType(feature_vector))

                feature = ROIFeature()
                feature.feature_type = ROIFeatureType(feature_vector)
                feature.feature_vector = vector[0].tolist()

                image.features.append(feature)

            roi.images.append(image)
        cv2.waitKey(0) # Debug

    # endregion


app = Learning()
app.learn()
