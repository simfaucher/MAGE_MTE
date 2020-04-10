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

from MLValidation import MLValidation
from VCLikeEngine import VCLikeEngine
from SIFTEngine import SIFTEngine

CAPTURE_DEMO = False
DEMO_FOLDER = "demo/"

# CAM_MATRIX = np.array([[954.16160543, 0., 635.29854945], \
#     [0., 951.09864051, 359.47108905],  \
#         [0., 0., 1.]])

VC_LIKE_ENGINE_MODE = False
SIFT_ENGINE_MODE = not VC_LIKE_ENGINE_MODE

class MTE:
    def __init__(self):
        print("Launching server")
        self.image_hub = imagezmq.ImageHub()
        self.image_hub.zmq_socket.RCVTIMEO = 3000
        # self.image_hub = imagezmq.ImageHub(open_port='tcp://192.168.43.39:5555')

        self.repo = Repository()

        self.learning_db = []
        self.last_learning_data = None

        # ML validation
        self.ml_validator = MLValidation()

        if CAPTURE_DEMO:
            self.out = None
            if not os.path.exists(DEMO_FOLDER):
                os.makedirs(DEMO_FOLDER)

        # Motion tracking engines
        self.sift_engine = SIFTEngine()
        self.vc_like_engine = VCLikeEngine()

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
        if SIFT_ENGINE_MODE:
            kp, _, _ = self.sift_engine.compute_sift(image, crop_image=True)
            return len(kp)

        return 0

    def learning(self, full_image):
        # Enregistrement de l'image de référence en 640 pour SIFT + VC léger et 4K pour VCE
        learning_id = self.repo.save_new_pov(full_image)

        success, learning_data = self.repo.get_pov_by_id(learning_id)
        if success:
            self.learning_db.append(learning_data)

        return learning_id

    def recognition(self, pov_id, image):
        # Récupération d'une image, SIFT puis si validé VC léger avec mires auto
        ret_data = {
            "scale": "OK",
            "skew": "OK",
            "translation": {
                "x": "OK",
                "y": "OK"
            },
            "success": False
        }

        learning_data = self.get_learning_data(pov_id)

        if VC_LIKE_ENGINE_MODE:
            success, scale, angle, transformed = self.vc_like_engine.find_target(image, learning_data)

            # cv2.imshow("VC-like engine", transformed)
        else:
            success, scale, skew, translation, transformed = self.sift_engine.recognition(image, learning_data)

        # ML validation
        ml_success = self.ml_validator.validate(learning_data, transformed)

        if not ml_success:
            # Scale
            if scale < SIFTEngine.HOMOGRAPHY_MIN_SCALE:
                ret_data["scale"] = "far"
            elif scale > SIFTEngine.HOMOGRAPHY_MAX_SCALE:
                ret_data["scale"] = "close"

            #TODO: à modifier en prenant en compte les infos de VC-like
            if SIFT_ENGINE_MODE:
                # Skew
                if -1*SIFTEngine.HOMOGRAPHY_MAX_SKEW > skew:
                    ret_data["skew"] = "minus"
                elif skew > SIFTEngine.HOMOGRAPHY_MAX_SKEW:
                    ret_data["skew"] = "plus"

                # Translation
                if translation[0] < SIFTEngine.HOMOGRAPHY_MIN_TRANS:
                    ret_data["translation"]["x"] = "minus"
                elif translation[0] > SIFTEngine.HOMOGRAPHY_MAX_TRANS:
                    ret_data["translation"]["x"] = "plus"

                if translation[1] < SIFTEngine.HOMOGRAPHY_MIN_TRANS:
                    ret_data["translation"]["y"] = "minus"
                elif translation[1] > SIFTEngine.HOMOGRAPHY_MAX_TRANS:
                    ret_data["translation"]["y"] = "plus"
            else:
                pass

        # if CAPTURE_DEMO:
        #     if self.out is None:
        #         h_matching, w_matching = matching_result.shape[:2]

        #         demo_path = os.path.join(DEMO_FOLDER, 'demo_recognition_{}.avi'.format(int(round(time.time() * 1000))))
        #         self.out = cv2.VideoWriter(demo_path, \
        #             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, \
        #             (w_matching, h_matching))

        #     self.out.write(matching_result)

        cv2.imshow("Transformed", transformed)
        cv2.waitKey(1)

        ret_data["success"] = success and ml_success

        return success, ret_data

    def framing(self, pov_id, image):
        # Recadrage avec SIFT et renvoi de l'image
        if SIFT_ENGINE_MODE:
            learning_data = self.get_learning_data(pov_id)

            sift_success, src_pts, dst_pts = self.sift_engine.apply_sift(image, learning_data.sift_data)

            if sift_success:
                h, w = image.shape[:2]
                H = self.sift_engine.get_homography_matrix(src_pts, dst_pts, dst_to_src=True)
                warped_image = cv2.warpPerspective(image, H, (w, h))
                return sift_success, warped_image
            else:
                return sift_success, image
        else:
            #TODO: à faire
            return False, image


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

        # Learn VC-like engine data
        self.vc_like_engine.learn(learning_data)

        # Learn SIFT data
        self.sift_engine.learn(learning_data)

        # Learn ML data
        self.ml_validator.learn(learning_data)

        # cv2.waitKey(0)
        return learning_data

if __name__ == "__main__":
    mte = MTE()
    mte.listen_images()
