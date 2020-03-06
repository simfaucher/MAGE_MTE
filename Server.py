"""
    Server side to catch a camera stream from a client
"""

import os
import sys
import time
from copy import deepcopy
import json
import numpy as np
import cv2
from pykson import Pykson
import imagezmq

from Domain.MTEMode import MTEMode
from Domain.LearningData import LearningData
from Domain.SiftData import SiftData
from Domain.MLData import MLData
from Repository import Repository

from ML.Domain.LearningKnowledge import LearningKnowledge
from ML.Domain.Image import Image
from ML.Domain.ROIFeatureType import ROIFeatureType
from ML.Domain.ROIFeature import ROIFeature
from ML.Domain.ImageFilterType import ImageFilterType
from ML.Domain.Point2D import Point2D
from ML.Domain.ImageClass import ImageClass

from ML.LinesDetector import LinesDetector
from ML.BoxLearner import BoxLearner

CROP_SIZE_HOR = 1/3
CROP_SIZE_VER = 1/3

FLANN_INDEX_KDTREE = 0
INDEX_PARAMS = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

SEARCH_PARAMS = dict(checks=50)
FLANN_THRESH = 0.7
MIN_MATCH_COUNT = 30

HOMOGRAPHY_MIN_SCALE = 0.75
HOMOGRAPHY_MAX_SCALE = 1.25
HOMOGRAPHY_MAX_SKEW = 0.13
HOMOGRAPHY_MIN_TRANS = 0
HOMOGRAPHY_MAX_TRANS = 50

CONFIG_SIGHTS_FILENAME = "learning_settings.json"

CAPTURE_DEMO = False
DEMO_FOLDER = "demo/"

# CAM_MATRIX = np.array([[954.16160543, 0., 635.29854945], \
#     [0., 951.09864051, 359.47108905],  \
#         [0., 0., 1.]])

class MTE:
    def __init__(self):
        print("Launching server")
        self.image_hub = imagezmq.ImageHub()
        self.image_hub.zmq_socket.RCVTIMEO = 3000
        # self.image_hub = imagezmq.ImageHub(open_port='tcp://192.168.43.39:5555')

        self.repo = Repository()
        self.sift = cv2.xfeatures2d.SIFT_create()

        self.learning_db = []
        self.last_learning_data = None

        self.load_ml_settings()
        self.box_learner = BoxLearner(self.learning_settings.sights, \
            self.learning_settings.recognition_selector.uncertainty)

        if CAPTURE_DEMO:
            self.out = None
            if not os.path.exists(DEMO_FOLDER):
                os.makedirs(DEMO_FOLDER)

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

    def listen_images(self):
        while True:  # show streamed images until Ctrl-C
            msg, image = self.image_hub.recv_image()

            data = json.loads(msg)

            ret_data = {}

            if "error" in data and data["error"]:
                if CAPTURE_DEMO and self.out is not None:
                    print("No connection")
                    self.out.release()
                    self.out = None
                    cv2.destroyWindow("Matching result")
                continue

            mode = MTEMode(data["mode"])
            if mode == MTEMode.PRELEARNING:
                print("MODE prelearning")
                nb_kp = self.prelearning(image)
                # save_ref = "save_ref" in data and data["save_ref"]
                # ret_data["prelearning_pts"] = self.get_rectangle(0, image, force_new_ref=save_ref)
                ret_data["prelearning"] = {
                    "nb_kp": nb_kp
                }
            elif mode == MTEMode.LEARNING:
                print("MODE learning")
                learning_id = self.learning(image)

                ret_data["learning"] = {
                    "id": learning_id
                }
            elif mode == MTEMode.RECOGNITION:
                pov_id = data["pov_id"]
                # print("MODE recognition")
                success, recog_ret_data = self.recognition(pov_id, image)

                ret_data["recognition"] = recog_ret_data
                ret_data["recognition"]["success"] = success
            # elif mode == MTEMode.FRAMING:
            else:
                pov_id = data["pov_id"]
                print("MODE framing")
                success, warped_image = self.framing(pov_id, image)

                ret_data["framing"] = {
                    "success": success
                }

                # cv2.imshow("Warped image", warped_image)
                # cv2.waitKey(1)
            
            if mode == MTEMode.FRAMING:
                self.image_hub.send_reply_image(warped_image, json.dumps(ret_data))
            else:
                self.image_hub.send_reply(json.dumps(ret_data).encode())

    def prelearning(self, image):
        # Renvoyer le nombre d'amers sur l'image envoyée
        img = self.crop_image(image)
        kp, _ = self.sift.detectAndCompute(img, None)
        return len(kp)

    def learning(self, full_image):
        # Enregistrement de l'image de référence en 640 pour SIFT + VC léger et 4K pour VCE
        learning_id = self.repo.save_new_pov(full_image)

        success, learning_data = self.repo.get_pov_by_id(learning_id)
        if success:
            self.learning_db.append(learning_data)

        return learning_id

    def recognition(self, pov_id, image):
        # Récupération d'une image, SIFT puis si validé VC léger avec mires auto
        ret_data = {}

        learning_data = self.get_learning_data(pov_id)

        sift_success, src_pts, dst_pts, kp_img, des_img, good_matches = self.apply_sift(image, learning_data, debug=True)

        if sift_success:
            H, mask = self.get_homography_matrix(src_pts, dst_pts, return_mask=True)
            matches_mask = mask.ravel().tolist()

            h, w = learning_data.sift_data.ref.shape[:2]
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H)

            debug_img = cv2.polylines(image.copy(), [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

            # cv2.imshow("Deformation", debug_img)

            scale_x = H[0][0]
            scale_y = H[1][1]
            skew_x = H[0][1]
            skew_y = H[1][0]
            t_x = H[0][2]
            t_y = H[1][2]

            scale_ok = HOMOGRAPHY_MIN_SCALE <= scale_x <= HOMOGRAPHY_MAX_SCALE \
                and HOMOGRAPHY_MIN_SCALE <= scale_y <= HOMOGRAPHY_MAX_SCALE
            skew_ok = 0 <= abs(skew_x) <= HOMOGRAPHY_MAX_SKEW \
                and 0 <= abs(skew_y) <= HOMOGRAPHY_MAX_SKEW
            translation_ok = HOMOGRAPHY_MIN_TRANS <= t_x <= HOMOGRAPHY_MAX_TRANS \
                and HOMOGRAPHY_MIN_TRANS <= t_y <= HOMOGRAPHY_MAX_TRANS

            ml_success = False
            if scale_ok and skew_ok and translation_ok:
                print("Valide")

                # Framing
                H = self.get_homography_matrix(src_pts, dst_pts, dst_to_src=True)
                warped_image = cv2.warpPerspective(image, H, (w, h))

                cv2.imshow("Warped image", warped_image)

                # ML validation
                ml_success = self.ml_validation(learning_data, warped_image)

            if not ml_success:
                ret_data["scale"] = "OK"
                ret_data["skew"] = {
                    "x": "OK",
                    "y": "OK"
                }
                ret_data["translation"] = {
                    "x": "OK",
                    "y": "OK"
                }

                # Scale
                if scale_x < HOMOGRAPHY_MIN_SCALE or scale_y < HOMOGRAPHY_MIN_SCALE:
                    ret_data["scale"] = "far"
                elif scale_x > HOMOGRAPHY_MAX_SCALE or scale_y > HOMOGRAPHY_MAX_SCALE:
                    ret_data["scale"] = "close"

                # Skew
                if -1*HOMOGRAPHY_MAX_SKEW > skew_x:
                    ret_data["skew"]["x"] = "minus"
                elif skew_x > HOMOGRAPHY_MAX_SKEW:
                    ret_data["skew"]["x"] = "plus"

                if -1*HOMOGRAPHY_MAX_SKEW > skew_y:
                    ret_data["skew"]["y"] = "minus"
                elif skew_y > HOMOGRAPHY_MAX_SKEW:
                    ret_data["skew"]["y"] = "plus"

                # Translation
                if t_x < HOMOGRAPHY_MIN_TRANS:
                    ret_data["translation"]["x"] = "minus"
                elif t_x > HOMOGRAPHY_MAX_TRANS:
                    ret_data["translation"]["x"] = "plus"

                if t_y < HOMOGRAPHY_MIN_TRANS:
                    ret_data["translation"]["y"] = "minus"
                elif t_y > HOMOGRAPHY_MAX_TRANS:
                    ret_data["translation"]["y"] = "plus"
        else:
            matches_mask = None
            debug_img = image.copy()

        DRAW_PARAMS = dict(matchColor=(0, 255, 0), \
                        singlePointColor=(255, 0, 0), \
                        matchesMask=matches_mask, \
                        flags=0)

        matching_result = cv2.drawMatches(debug_img, kp_img, learning_data.sift_data.ref, learning_data.sift_data.kp, good_matches, None, **DRAW_PARAMS)
        cv2.imshow("Matching result", matching_result)

        if CAPTURE_DEMO:
            if self.out is None:
                h_matching, w_matching = matching_result.shape[:2]

                demo_path = os.path.join(DEMO_FOLDER, 'demo_recognition_{}.avi'.format(int(round(time.time() * 1000))))
                self.out = cv2.VideoWriter(demo_path, \
                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, \
                    (w_matching, h_matching))

            self.out.write(matching_result)

        cv2.waitKey(1)

        ret_data["sift_success"] = sift_success

        success = sift_success and ml_success

        return success, ret_data

    def ml_validation(self, learning_data, warped_image):
        success = len(learning_data.ml_data.sights) > 0

        for sight in learning_data.ml_data.sights:
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

        return success

    def framing(self, pov_id, image):
        # Recadrage avec SIFT et renvoi de l'image
        learning_data = self.get_learning_data(pov_id)

        sift_success, src_pts, dst_pts = self.apply_sift(image, learning_data)

        if sift_success:
            h, w = image.shape[:2]
            H = self.get_homography_matrix(src_pts, dst_pts, dst_to_src=True)
            warped_image = cv2.warpPerspective(image, H, (w, h))
            return sift_success, warped_image
        else:
            return sift_success, image

    def crop_image(self, image):
        h, w = image.shape[:2]
        croped = image[int(h*CROP_SIZE_VER/2): int(h*(1-CROP_SIZE_VER/2)), \
            int(w*CROP_SIZE_HOR/2): int(w*(1-CROP_SIZE_HOR/2))]

        return croped

    def get_learning_data(self, pov_id):
        learning_data = None

        if self.last_learning_data is not None \
            and self.last_learning_data.id == pov_id:
            learning_data = self.last_learning_data
        else:
            items = [x for x in self.learning_db if x.id == pov_id]
            if len(items) > 0:
                learning_data = items[0]
            else:
                success, learning_data = self.repo.get_pov_by_id(pov_id)
                if not success:
                    raise Exception("No POV with id {}".format(pov_id))

        self.last_learning_data = learning_data

        # Learn SIFT data
        if learning_data.sift_data is None:
            croped = self.crop_image(learning_data.image_640)
            kp, des = self.sift.detectAndCompute(croped, None)

            learning_data.sift_data = SiftData(kp, des, croped)

        # Learn ML data
        if learning_data.ml_data is None:
            learning_data.ml_data = deepcopy(self.learning_settings)

            image_class = ImageClass()
            image_class.id = 0
            image_class.name = "Reference"

            h, w = learning_data.image_640.shape[:2]

            for sight in learning_data.ml_data.sights:
                pt_tl = Point2D()
                pt_tl.x = int(w / 2 - sight.width / 2)
                pt_tl.y = int(h / 2 - sight.height / 2)

                pt_br = Point2D()
                pt_br.x = pt_tl.x + sight.width
                pt_br.y = pt_tl.y + sight.height

                sight_image = learning_data.image_640[pt_tl.y: pt_br.y, pt_tl.x: pt_br.x]
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

        # cv2.waitKey(0)
        return learning_data

    def get_homography_matrix(self, src_pts, dst_pts, dst_to_src=False, return_mask=False):
        if dst_to_src:
            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        else:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if return_mask:
            return H, mask
        else:
            return H

    def apply_sift(self, image, learning_data, crop_image=True, debug=False):
        h, w = image.shape[:2]

        if crop_image:
            img = self.crop_image(image)
        else:
            img = image

        # h_img, w_img = img.shape[:2]
        kp_img, des_img = self.sift.detectAndCompute(img, None)

        flann = cv2.FlannBasedMatcher(INDEX_PARAMS, SEARCH_PARAMS)

        matches = flann.knnMatch(des_img, learning_data.sift_data.des, k=2)

        # Need to draw only good matches, so create a mask
        good_matches = []

        # ratio test as per Lowe's paper
        for i, pair in enumerate(matches):
            try:
                m, n = pair
                if m.distance < FLANN_THRESH*n.distance:
                    good_matches.append(m)
            except ValueError:
                pass

        # Add crop
        if crop_image:
            for kp in kp_img:
                kp.pt = (kp.pt[0] + w * CROP_SIZE_HOR/2, kp.pt[1] + h * CROP_SIZE_VER/2)

        # Homography
        # print("Matches found: %d/%d" % (len(goodMatches), MIN_MATCH_COUNT))

        success = len(good_matches) > MIN_MATCH_COUNT

        dst_pts = []
        src_pts = []
        if success:
            dst_pts = np.float32([kp_img[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            src_pts = np.float32([learning_data.sift_data.kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        if debug:
            return success, src_pts, dst_pts, kp_img, des_img, good_matches
        else:
            return success, src_pts, dst_pts


if __name__ == "__main__":
    mte = MTE()
    mte.listen_images()
