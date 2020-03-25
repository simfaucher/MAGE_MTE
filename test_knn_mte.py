import sys
import glob
import itertools
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

CONFIG_VALIDATION_SIGHTS_FILENAME = "learning_settings_validation.json"
CONFIG_VALIDATION_SIGHTS_2_FILENAME = "learning_settings_validation2.json"
ROTATION_IMAGES_FOLDER = "images/rotation/*"

class Test:
    def __init__(self):
        image_path = "videos/T1.1/vlcsnap-2020-03-02-15h59m47s327.png"
        image = cv2.imread(image_path)
        image = imutils.resize(image, width=640)

        self.learning_settings = self.load_ml_settings(CONFIG_VALIDATION_SIGHTS_FILENAME)
        self.box_learner = BoxLearner(self.learning_settings.sights, \
            self.learning_settings.recognition_selector.uncertainty)

        self.learning_settings2 = self.load_ml_settings(CONFIG_VALIDATION_SIGHTS_2_FILENAME)
        self.box_learner2 = BoxLearner(self.learning_settings2.sights, \
            self.learning_settings2.recognition_selector.uncertainty)

        self.dataset = self.generate_dataset(image)
        self.learn_ml_data()
        # self.learn_ml_data2()
        self.learn_ml_data2(100)

    def generate_dataset(self, image):
        h, w = image.shape[:2]

        dataset = []
        # Scale levels
        s1 = [0.5, 0.75, 0.85]
        s2 = np.arange(0.9, 1.1, 0.01)
        s3 = [1.15, 1.25, 1.5]
        scales = itertools.chain(s1, s2, s3)
        for scale in scales:
            scale = round(scale, 2)

            # Rotation levels
            a1 = range(-40, -11, 10)
            a2 = range(-10, 11, 1)
            a3 = range(20, 41, 10)
            angles = itertools.chain(a1, a2, a3)
            for angle in angles:
                scaled = self.scale_image(image, scale)
                M = cv2.getRotationMatrix2D(((w-1)/2.0, (h-1)/2.0), angle, 1)
                transformed = cv2.warpAffine(scaled, M, (w, h))

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

    def learn_ml_data(self):
        for i, data in enumerate(self.dataset):
            attr, image = data[:]
            class_id = int(attr["scale"]*100)
            class_name = "scale: {}".format(attr["scale"])

            self.learn_image(self.learning_settings, class_id, class_name, image)

    def learn_ml_data2(self, scale):
        # for sight in self.learning_settings2.sights:
        #     for j, roi in enumerate(sight.roi):
        #         roi.images = []

        for i, data in enumerate(self.dataset):
            attr, image = data[:]

            current_scale = int(attr["scale"]*100)

            if current_scale == scale:
                class_id = int(attr["angle"]) + 100
                class_name = "angle: {}".format(attr["angle"])

                self.learn_image(self.learning_settings2, class_id, class_name, image)

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
            box_learner.input_image = image

            h, w = image.shape[:2]

            pt_tl = Point2D()
            pt_tl.x = int(w / 2 - sight.width / 2)
            pt_tl.y = int(h / 2 - sight.height / 2)

            pt_br = Point2D()
            pt_br.x = pt_tl.x + sight.width
            pt_br.y = pt_tl.y + sight.height

            match = box_learner.find_target(pt_tl, pt_br)

            success = match.success if not match.success else success
            matches.append(match)

        return success, matches

    def scale_image(self, image, scale):
        if scale == 1:
            return image

        h, w, c = image.shape

        resized = cv2.resize(image, None, fx=scale, fy=scale)
        h_r, w_r = resized.shape[:2]

        margin_h = int(abs(h_r - h) / 2)
        margin_w = int(abs(w_r - w) / 2)
        if scale < 1:
            dest = np.zeros((h, w, c), dtype=image.dtype)
            dest[margin_h: margin_h + h_r, margin_w: margin_w + w_r] = resized
        else:
            dest = resized[margin_h: margin_h + h, margin_w: margin_w + w]

        return dest

if __name__ == "__main__":
    app = Test()

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

    scale_errors = {}
    angle_errors = {}

    fps = FPS().start()

    for data, image in dataset:
        h, w = image.shape[:2]

        # Scale
        success1, matches1 = app.ml_validation(app.learning_settings, app.box_learner, image)

        scale = 1
        angle = 0

        if success1:
            # scale = 1 + float(100 - matches1[0].predicted_class)/100
            scale = 1 / (matches1[0].predicted_class / 100)
            scale_error = abs(int(data["scale"]*100) - matches1[0].predicted_class)
            print("Scale success, class: {}, expected: {}, error: {}".format(matches1[0].predicted_class, \
                int(data["scale"]*100), scale_error))

            # Data for plotting
            scale_label = int(round(data["scale"]*100))
            if scale_label not in scale_errors:
                scale_errors[scale_label] = []

            scale_errors[scale_label].append(scale_error)

            M = cv2.getRotationMatrix2D(((w-1)/2.0, (h-1)/2.0), 0, scale)
            scaled = cv2.warpAffine(image, M, (w, h))

            # scaled = app.scale_image(image, scale)

            # cv2.imshow("Scaled", scaled)

            # Rotation
            success2, matches2 = app.ml_validation(app.learning_settings2, app.box_learner2, scaled)

            if success2:
                angle = -1 * (matches2[0].predicted_class - 100)
                angle_error = abs((matches2[0].predicted_class - 100) - data["angle"])

                # Data for plotting
                angle_label = int(round(data["angle"]))
                if angle_label not in angle_errors:
                    angle_errors[angle_label] = []

                angle_errors[angle_label].append(angle_error)

                print("Rotation success, class: {}, expected: {}, error: {}".format(matches2[0].predicted_class - 100, \
                    data["angle"], angle_error))
            else:
                print("Rotation fail")
        else:
            print("Scale fail")

        M = cv2.getRotationMatrix2D(((w-1)/2.0, (h-1)/2.0), angle, scale)
        transformed = cv2.warpAffine(image, M, (w, h))

        fps.update()

        # cv2.imshow("Image", image)
        # cv2.imshow("Transformed", transformed)
        # cv2.waitKey(100)
    
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # Plot angle errors
    plt.figure(1)
    angle_labels, angle_data = [*zip(*angle_errors.items())]
    plt.boxplot(angle_data)
    plt.xticks(range(1, len(angle_labels) + 1), angle_labels)
    plt.title("Angle errors")
    plt.xlabel("Angle (deg)")
    plt.ylabel("Error (deg)")

    # Plot scale errors
    plt.figure(2)
    scale_labels, scale_data = [*zip(*scale_errors.items())]
    plt.boxplot(scale_data)
    plt.xticks(range(1, len(scale_labels) + 1), scale_labels)
    plt.title("Scale errors")
    plt.xlabel("Scale (% of the original image)")
    plt.ylabel("Error (% of the original image)")

    plt.show()
