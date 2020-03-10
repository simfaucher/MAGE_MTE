import sys
import glob
import json
from pykson import Pykson
import cv2
import imutils

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

    def generate_dataset(self, image):
        h, w = image.shape[:2]

        dataset = []
        # Scale levels
        for scale in [0.75, 1, 1.25]:
            # Rotation levels
            for angle in range(-40, 41, 5):
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

    def learn_ml_data(self):
        for i, data in enumerate(self.dataset):
            attr, image = data[:]
            class_id = int(attr["scale"]*100)
            class_name = "scale: {}".format(attr["scale"])

            self.learn_image(self.learning_settings, class_id, class_name, image)

    def learn_ml_data2(self, scale):
        for sight in self.learning_settings2.sights:
            for j, roi in enumerate(sight.roi):
                roi.images = []

        for i, data in enumerate(self.dataset):
            attr, image = data[:]

            current_scale = int(attr["scale"]*100)

            if current_scale == scale:
                class_id = int(attr["angle"])
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

    def ml_validation(self, learning_settings, image):
        success = len(learning_settings.sights) > 0

        matches = []

        for sight in learning_settings.sights:
            self.box_learner.get_knn_contexts(sight)
            self.box_learner.input_image = image

            h, w = image.shape[:2]

            pt_tl = Point2D()
            pt_tl.x = int(w / 2 - sight.width / 2)
            pt_tl.y = int(h / 2 - sight.height / 2)

            pt_br = Point2D()
            pt_br.x = pt_tl.x + sight.width
            pt_br.y = pt_tl.y + sight.height

            match = self.box_learner.find_target(pt_tl, pt_br)

            success = match.success if not match.success else success
            matches.append(match)

        return success, matches

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

    for data, image in dataset:
        # Scale
        success1, matches1 = app.ml_validation(app.learning_settings, image)

        scale = 1
        angle = 0
        if success1:
            scale = 1 + float(100 - matches1[0].predicted_class)/100
            print("Scale success, class: {}".format(matches1[0].predicted_class))

            # Rotation
            app.learn_ml_data2(matches1[0].predicted_class)
            success2, matches2 = app.ml_validation(app.learning_settings2, image)

            if success2:
                angle = -1 * matches2[0].predicted_class
                print("Rotation success, class: {}".format(matches2[0].predicted_class))
            else:
                print("Rotation fail")
        else:
            print("Scale fail")

        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D(((w-1)/2.0, (h-1)/2.0), angle, scale)
        transformed = cv2.warpAffine(image, M, (w, h))

        cv2.imshow("Image", image)
        cv2.imshow("Transformed", transformed)
        cv2.waitKey(0)
