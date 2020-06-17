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
import imutils
from imutils.video import FPS
from pykson import Pykson
import imagezmq
import argparse
import csv

from Domain.MTEMode import MTEMode
from Domain.MTEAlgo import MTEAlgo
from Domain.LearningData import LearningData
from Domain.SiftData import SiftData
from Domain.MLData import MLData
from Repository import Repository
from Domain.MTEResponse import MTEResponse
from Domain.MTEThreshold import MTEThreshold
from Domain.RecognitionData import RecognitionData
from Domain.ResponseData import ResponseData

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
from D2NetEngine import D2NetEngine

CAPTURE_DEMO = False
DEMO_FOLDER = "demo/"

# CAM_MATRIX = np.array([[954.16160543, 0., 635.29854945], \
#     [0., 951.09864051, 359.47108905],  \
#         [0., 0., 1.]])

VC_LIKE_ENGINE_MODE = False
SIFT_ENGINE_MODE = not VC_LIKE_ENGINE_MODE

class MTE:
    def __init__(self, mte_algo=MTEAlgo.SIFT_KNN, crop_margin=1.0/6, resize_width=640, ransacount=300):
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
        self.mte_algo = mte_algo
        self.crop_margin = crop_margin
        self.validation_width = 380
        self.validation_height = int((self.validation_width/16)*9)
        self.resize_width = resize_width
        self.resize_height = int((resize_width/16)*9)

        if self.mte_algo in (MTEAlgo.D2NET_KNN, MTEAlgo.D2NET_RANSAC):
            self.d2net_engine = D2NetEngine(max_edge=resize_width, \
                                            max_sum_edges=resize_width + self.resize_height,\
                                            maxRansac=ransacount, width=self.resize_width, \
                                            height=self.resize_height)
        else:
            self.sift_engine = SIFTEngine(maxRansac=ransacount, width=self.validation_width, height=self.validation_height)

        self.threshold_380 = MTEThreshold(260, 45, 2900, 900, 10000, 4000, 11000)
        self.threshold_640 = MTEThreshold(750, 70, 2700, 1050, 12000, 5000, 15000)
        self.threshold_1728 = MTEThreshold(3500, 180, 3100, 750, 13000, 5500, 20000)

        self.rollback = 0
        self.validation = 0
        self.devicetype = "CPU"
        self.resolution_change_allowed = 3
        

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

            image_for_learning = image

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
                learning_data = self.learning(image_for_learning)
                if learning_data["success"]:
                    ret_data["learning"] = {"id" : learning_data["learning_id"]}

                # TODO : convert keypoints type into float array if we want to save them
                # ret_data["learning"] = learning_data
            elif mode == MTEMode.RECOGNITION:
                pov_id = data["pov_id"]
                # print("MODE recognition")
                if self.devicetype == "CPU" and image.shape[0] > 640:
                    image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
                results = RecognitionData(*self.recognition(pov_id, image))

                ret_data["recognition"] = results.recog_ret_data
                ret_data["recognition"]["success"] = results.success
                if image.shape[0] == 380:
                    response_for_client = self.behaviour_380(results)
                elif image.shape[0] == 640:
                    response_for_client = self.behaviour_640(results)
                else:
                    response_for_client = self.behaviour_1728(results)
                if self.validation > 5:
                    self.validation = 5
                if self.validation == 5:
                    is_blurred = self.is_image_blurred(image, \
                        size=int(response_for_client.size/18), thresh=10)
                    if not is_blurred[1]:
                        response_for_client.response = MTEResponse.CAPTURE
                        self.validation = 0
                        self.rollback = 0
                ret_data["recognition"]["results"] = response_for_client.convert_to_dict()
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

    def is_image_blurred(self, image, size=60, thresh=10):
        (h, w) = image.shape
        (cX, cY) = (int(w / 2.0), int(h / 2.0))
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)
        fft_shift[cY - size:cY + size, cX - size:cX + size] = 0
        fft_shift = np.fft.ifftshift(fft_shift)
        recon = np.fft.ifft2(fft_shift)
        magnitude = 20 * np.log(np.abs(recon))
        mean = np.mean(magnitude)
        return (mean, mean <= thresh)

    def compute_direction(self,translation,size):
        #size, t_x, t_y peuvent etre récup avec le self
        return "H G"

    def red_380(self):
        size = 380
        self.rollback += 1
        if self.rollback >= 5:
            self.rollback = 0
            size = 640
        return ResponseData(size, MTEResponse.RED, None, None, None, None, None)

    def orange_behaviour(self, results, size):
        if self.validation > 0:
            self.validation -= 1
        if size == 640 and self.rollback > 0:
            self.rollback -= 1

        return ResponseData(size, MTEResponse.ORANGE,\
                            results.translations[0], results.translations[1], \
                            self.compute_direction(results.translations, size), \
                            results.scales[0], results.scales[1])

    def red_640(self):
        size = 640
        msg = MTEResponse.RED
        if self.devicetype == "CPU":
            msg = MTEResponse.TARGET_LOST
        else:
            self.rollback += 1
            if self.rollback >= 5:
                size = 1728
                self.rollback = 0
        return ResponseData(size, msg, None, None, None, None, None)

    def green_640(self, results):
        size = 640
        if self.resolution_change_allowed > 0:
            self.resolution_change_allowed -= 1
            size = 380
            self.validation = 0
            self.rollback = 0
        else:
            self.validation += 1
        return ResponseData(size, MTEResponse.GREEN,\
                            results.translations[0], results.translations[1], \
                            self.compute_direction(results.translations, 640), \
                            results.scales[0], results.scales[1])

    def lost_1728(self):
        msg = MTEResponse.TARGET_LOST
        return ResponseData(1728, msg, None, None, None, None, None)

    def behaviour_380(self, results):
        response = MTEResponse.RED
        # If not enough keypoints
        if results.nb_kp < self.threshold_380.nb_kp:
            response_for_client = self.red_380()
        # If not enough matches
        elif results.nb_match < self.threshold_380.nb_match:
            # If homography doesn't even start
            if results.nb_match < 30:
                response_for_client = self.red_380()
            else:
                response_for_client = self.orange_behaviour(results, 380)
        else:
            response = MTEResponse.GREEN
            # If not centered with target
            if not results.success:
                self.validation = 0
                response_for_client = ResponseData(380, response,\
                                     results.translations[0], results.translations[1], \
                                     self.compute_direction(results.translations, 380), \
                                     results.scales[0], results.scales[1])
            else:
                dist_kirsh = results.dist_roi[0] < self.threshold_380.mean_kirsh
                dist_canny = results.dist_roi[1] < self.threshold_380.mean_canny
                dist_color = results.dist_roi[2] < self.threshold_380.mean_color
                # If 0 or 1 mean valid
                if dist_kirsh+dist_canny+dist_color < 2:
                    response_for_client = self.orange_behaviour(results, 380)
                else:
                    dist_kirsh = results.dist_roi[0] < self.threshold_380.kirsh_aberration
                    dist_color = results.dist_roi[2] < self.threshold_380.color_aberration
                    # If 0 aberration
                    if dist_kirsh+dist_color == 2:
                        self.validation += 1
                        self.rollback = 0
                    else:
                        response = MTEResponse.ORANGE
                    response_for_client = ResponseData(380, response,\
                                     results.translations[0], results.translations[1], \
                                     self.compute_direction(results.translations, 380), \
                                     results.scales[0], results.scales[1])

        if response_for_client.response == MTEResponse.GREEN:
            self.rollback = 0
        return response_for_client

    def behaviour_640(self, results):
        if results.nb_kp < self.threshold_640.nb_kp:
            response_for_client = self.red_640()
        # If not enough matches
        elif results.nb_match < self.threshold_640.nb_match:
            # If homography doesn't even start
            if results.nb_match < 30:
                response_for_client = self.red_640()
            else:
                response_for_client = self.orange_behaviour(results, 640)
        else:
            response = MTEResponse.GREEN
            # If not centered with target
            if not results.success:
                response_for_client = ResponseData(640, response,\
                                     results.translations[0], results.translations[1], \
                                     self.compute_direction(results.translations, 640), \
                                     results.scales[0], results.scales[1])
            else:
                dist_kirsh = results.dist_roi[0] < self.threshold_640.mean_kirsh
                dist_canny = results.dist_roi[1] < self.threshold_640.mean_canny
                dist_color = results.dist_roi[2] < self.threshold_640.mean_color
                # If 0 or 1 mean valid
                if dist_kirsh+dist_canny+dist_color < 2:
                    response_for_client = self.orange_behaviour(results, 640)
                # If all means are valids
                elif dist_kirsh+dist_canny+dist_color == 3:
                    response_for_client = self.green_640(results)
                else:
                    dist_kirsh = results.dist_roi[0] < self.threshold_380.kirsh_aberration
                    dist_color = results.dist_roi[2] < self.threshold_380.color_aberration
                    # If 0 aberration
                    if dist_kirsh+dist_color == 2:
                        response_for_client = self.green_640(results)
                    else:
                        response_for_client = self.orange_behaviour(results, 640)

        if response_for_client.response == MTEResponse.GREEN:
            self.rollback = 0
        return response_for_client

    def behaviour_1728(self, results):
        if results.nb_kp < self.threshold_1728.nb_kp:
            response_for_client = self.lost_1728()
        # If not enough matches
        elif results.nb_match < self.threshold_1728.nb_match:
            response_for_client = self.lost_1728()
        else:
            response = MTEResponse.GREEN
            # If not centered with target
            if not results.success:
                response_for_client = ResponseData(1728, response,\
                                     results.translations[0], results.translations[1], \
                                     self.compute_direction(results.translations, 1728), \
                                     results.scales[0], results.scales[1])
            else:
                dist_kirsh = results.dist_roi[0] < self.threshold_1728.mean_kirsh
                dist_canny = results.dist_roi[1] < self.threshold_1728.mean_canny
                dist_color = results.dist_roi[2] < self.threshold_1728.mean_color
                # If 0 or 1 mean valid
                if dist_kirsh+dist_canny+dist_color < 2:
                    response_for_client = self.lost_1728()
                # If all means are valids
                elif dist_kirsh+dist_canny+dist_color == 3:
                    response_for_client = ResponseData(640, response,\
                                     results.translations[0], results.translations[1], \
                                     self.compute_direction(results.translations, 1728), \
                                     results.scales[0], results.scales[1])
                else:
                    dist_kirsh = results.dist_roi[0] < self.threshold_1728.kirsh_aberration
                    dist_color = results.dist_roi[2] < self.threshold_1728.color_aberration
                    # If 0 aberration
                    if dist_kirsh+dist_color == 2:
                        size = 640
                    else:
                        response = MTEResponse.ORANGE
                        size = 1728
                    response_for_client = ResponseData(size, response,\
                                     results.translations[0], results.translations[1], \
                                     self.compute_direction(results.translations, 1728), \
                                     results.scales[0], results.scales[1])
        return response_for_client

    ### Iniatialize learning datas with the reference and avoid the DB's use
    ### In :    image_ref_reduite -> int array of the reduced reference
    ###         image_ref -> int array of the reference at full size
    def fake_init_for_reference(self, image_ref_reduite, image_ref):
        learning_data = LearningData(-1, "0", image_ref_reduite, image_ref)

        if self.mte_algo in (MTEAlgo.SIFT_KNN, MTEAlgo.SIFT_RANSAC):
            self.sift_engine.learn(learning_data, crop_image=True, crop_margin=self.crop_margin)
        elif self.mte_algo in (MTEAlgo.D2NET_KNN, MTEAlgo.D2NET_RANSAC):
            self.d2net_engine.learn(learning_data, crop_image=True, crop_margin=self.crop_margin)
        else:
            self.vc_like_engine.learn(learning_data)
        self.ml_validator.learn(learning_data)
        self.last_learning_data = learning_data

    ### Test the recognition between the input and the image learned with fakeInitForReference
    ### In :    blurred_image -> int array of the blurred reference
    ### Out :   results -> RecognitionData containing the recognition's results
    def test_filter(self, blurred_image):
        dim = (self.validation_width, self.validation_height)
        gaussian_redux = cv2.resize(blurred_image, dim, interpolation=cv2.INTER_AREA)
        results = RecognitionData(*self.recognition(-1, gaussian_redux))
        
        return results

    ### Check if the images in a folder are valid for MTE
    ### In  : image_ref -> int array of the potential reference
    ### Out : validation_value -> dictionnary indicating the success or failure of the image
    ### as well as the keypoints and theirs descriptors for severals dimensions of the image
    def check_reference(self, image_ref):
        kernel_size = 25
        sigma = 5
        kernel = 31

        kernel_v = np.zeros((kernel_size, kernel_size))
        kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
        kernel_v /= kernel_size

        kernel_h = np.zeros((kernel_size, kernel_size))
        kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
        kernel_h /= kernel_size

        # Compute keypoints for reference
        image_ref_kp, image_ref_desc = self.sift_engine.sift.detectAndCompute(image_ref, None)
        for i in range(len(image_ref_kp)):
            image_ref_kp[i].size = 2

        # Resize reference and compute keypoints
        dim = (self.validation_width, self.validation_height)
        image_ref_reduite = cv2.resize(image_ref, dim, interpolation=cv2.INTER_AREA)
        image_ref_reduite_kp, image_ref_reduite_desc = self.sift_engine.sift.detectAndCompute(image_ref_reduite, None)
        for i in range(len(image_ref_reduite_kp)):
            image_ref_reduite_kp[i].size = 1

        self.fake_init_for_reference(image_ref_reduite, image_ref)
        validation_value = {'success' : False}

        # Gaussian noise
        image_gaussian_blur = cv2.GaussianBlur(image_ref, (kernel, kernel), sigma)
        results = self.test_filter(image_gaussian_blur)
        if not results.success:
            print("Failure gaussian blur")
            return validation_value

        # Vertical motion blur.
        image_vertical_motion_blur = cv2.filter2D(image_ref, -1, kernel_v)
        results = self.test_filter(image_vertical_motion_blur)

        if not results.success:
            print("Failure vertical blur")
            return validation_value

        # Horizontal motion blur.
        image_horizontal_motion_blur = cv2.filter2D(image_ref, -1, kernel_h)
        results = self.test_filter(image_horizontal_motion_blur)

        if not results.success:
            print("Failure horizontal blur")
            return validation_value

        # The intermediary resolution
        dim = (640, 360)
        image_ref_half_redux = cv2.resize(image_ref, dim, interpolation=cv2.INTER_AREA)
        image_ref_half_redux_kp, image_ref_half_redux_desc = self.sift_engine.sift.detectAndCompute(image_ref_half_redux, None)
        for i in range(len(image_ref_half_redux_kp)):
            image_ref_half_redux_kp[i].size = 1
        # All 3 noises are valid
        validation_value = {'success' : True,
                            'full' : {'kp' : image_ref_kp, 
                                      'desc' : image_ref_desc
                                      },
                            'redux' : {'kp' : image_ref_reduite_kp,
                                       'desc' : image_ref_reduite_desc
                                       },
                            'half_redux' : {'kp' : image_ref_half_redux_kp,
                                            'desc' : image_ref_half_redux_desc
                                            }
        }
        print("Référence valide.")

        return validation_value

    def prelearning(self, image):
        # Renvoyer le nombre d'amers sur l'image envoyée
        if self.mte_algo in (MTEAlgo.SIFT_KNN, MTEAlgo.SIFT_RANSAC):
            kp, _, _,_ = self.sift_engine.compute_sift(image, crop_image=False)
            return len(kp)
        elif self.mte_algo in (MTEAlgo.D2NET_KNN, MTEAlgo.D2NET_RANSAC):
            kp, _, _,_ = self.d2net_engine.compute_d2(image, crop_image=True)
            return len(kp)

        return 0

    def learning(self, full_image):
        validation = self.check_reference(full_image)
        if validation["success"]:
            # Enregistrement de l'image de référence en 640 pour SIFT + VC léger et 4K pour VCE
            validation["learning_id"] = self.repo.save_new_pov(full_image)

            validation["success"], validation["learning_data"] = self.repo.get_pov_by_id(validation["learning_id"], resize_width=self.validation_width)
            if validation["success"]:
                self.learning_db.append(validation["learning_data"])                

        return validation

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
        sum_distances = 9999
        distances = 9999
        learning_data = self.get_learning_data(pov_id)

        fps = FPS().start()
        nb_matches = 0
        if self.mte_algo == MTEAlgo.VC_LIKE:
            success, scale, skew, transformed = self.vc_like_engine.find_target(image, learning_data)
            # cv2.imshow("VC-like engine", transformed)
        elif self.mte_algo in (MTEAlgo.D2NET_KNN, MTEAlgo.D2NET_RANSAC):
            success, scales, skews, translation, transformed, nb_matches, nb_kp = self.d2net_engine.recognition(image, learning_data, self.mte_algo)
            scale_x, scale_y = scales
            skew_x, skew_y = skews
            scale = max(scale_x, scale_y)
            skew = max(skew_x, skew_y)
        else:
            success, scales, skews, translation, transformed, nb_matches, nb_kp = self.sift_engine.recognition(image, learning_data, self.mte_algo)
            scale_x, scale_y = scales
            skew_x, skew_y = skews
            scale = max(scale_x, scale_y)
            skew = max(skew_x, skew_y)

        # ML validation
        ml_success = False
        if success:
            ml_success, sum_distances, distances = self.ml_validator.validate(learning_data, transformed)

        fps.update()
        fps.stop()

        if not ml_success:
            # Scale
            if scale < SIFTEngine.HOMOGRAPHY_MIN_SCALE:
                ret_data["scale"] = "far"
            elif scale > SIFTEngine.HOMOGRAPHY_MAX_SCALE:
                ret_data["scale"] = "close"

            #TODO: à modifier en prenant en compte les infos de VC-like
            if self.mte_algo == MTEAlgo.SIFT_KNN or self.mte_algo == MTEAlgo.SIFT_RANSAC:
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


        cv2.putText(transformed, "{:.2f} FPS".format(fps.fps()), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
            (255, 255, 255), 2)
        if self.mte_algo != MTEAlgo.VC_LIKE:
            cv2.putText(transformed, "{} matches".format(nb_matches), (160, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                (255, 255, 255), 2)
        if success:
            if self.mte_algo != MTEAlgo.VC_LIKE:
                cv2.putText(transformed, "Rot. x: {:.2f}".format(skew_x), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                (255, 255, 255), 2)
                cv2.putText(transformed, "Rot. y: {:.2f}".format(skew_y), (160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                (255, 255, 255), 2)
                cv2.putText(transformed, "Trans. x: {:.2f}".format(translation[0]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                (255, 255, 255), 2)
                cv2.putText(transformed, "Trans. y: {:.2f}".format(translation[1]), (160, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                (255, 255, 255), 2)
                cv2.putText(transformed, "Scale x: {:.2f}".format(scale_y), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                (255, 255, 255), 2)
                cv2.putText(transformed, "Scale y: {:.2f}".format(scale_x), (160, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                (255, 255, 255), 2)
            cv2.putText(transformed, "Dist.: {:.2f}".format(sum_distances), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
            (255, 255, 255), 2)

        cv2.imshow("Transformed", transformed)
        cv2.waitKey(1)

        ret_data["success"] = success and ml_success

        return success, ret_data, nb_kp, nb_matches, sum(translation),\
        sum(skews), sum_distances, distances, transformed, scales, translation

    def framing(self, pov_id, image):
        # Recadrage avec SIFT et renvoi de l'image
        if self.mte_algo in (MTEAlgo.SIFT_KNN, MTEAlgo.SIFT_RANSAC):
            learning_data = self.get_learning_data(pov_id)

            sift_success, src_pts, dst_pts, _ = self.sift_engine.apply_sift(image, learning_data.sift_data,self.mte_algo)

            if sift_success:
                h, w = image.shape[:2]
                H = self.sift_engine.get_homography_matrix(src_pts, dst_pts, dst_to_src=True)
                warped_image = cv2.warpPerspective(image, H, (w, h))
                return sift_success, warped_image
            else:
                return sift_success, image
        elif self.mte_algo in (MTEAlgo.D2NET_KNN, MTEAlgo.D2NET_RANSAC):
            learning_data = self.get_learning_data(pov_id)

            d2_success, src_pts, dst_pts, _ = self.d2net_engine.apply_d2(image, learning_data.sift_data,self.mte_algo)

            if d2_success:
                h, w = image.shape[:2]
                H = self.d2net_engine.get_homography_matrix(src_pts, dst_pts, dst_to_src=True)
                warped_image = cv2.warpPerspective(image, H, (w, h))
                return d2_success, warped_image
            else:
                return d2_success, image
        else:
            #TODO VCL
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
                success, learning_data = self.repo.get_pov_by_id(pov_id, resize_width=self.validation_width)
                if not success:
                    raise Exception("No POV with id {}".format(pov_id))

        self.last_learning_data = learning_data

        # Update : we only use 1 engine at a time
        if self.mte_algo in (MTEAlgo.SIFT_KNN, MTEAlgo.SIFT_RANSAC):
            # Learn SIFT data
            self.sift_engine.learn(learning_data, crop_image=True, crop_margin=self.crop_margin)
        elif self.mte_algo in (MTEAlgo.D2NET_KNN, MTEAlgo.D2NET_RANSAC):
            self.d2net_engine.learn(learning_data, crop_image=True, crop_margin=self.crop_margin)
        else:
            # Learn VC-like engine data
            self.vc_like_engine.learn(learning_data)

        # Learn ML data
        self.ml_validator.learn(learning_data)

        # cv2.waitKey(0)
        return learning_data

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'oui', 'o', 'vrai', 'v'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'non', 'faux'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    def convert_to_float(frac_str):
        try:
            return float(frac_str)
        except ValueError:
            num, denom = frac_str.split('/')
            try:
                leading, num = num.split(' ')
                whole = float(leading)
            except ValueError:
                whole = 0
            frac = float(num) / float(denom)
            return whole - frac if whole < 0 else whole + frac

    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--algo", required=False, default="SIFT_KNN",\
        help="Feature detection algorithm (SIFT_KNN, SIFT_RANSAC, D2NET_KNN, D2NET_RANSAC or VC_LIKE). Default: SIFT_KNN")
    ap.add_argument("-c", "--crop", required=False, default="1/6",\
        help="Part to crop around the center of the image (1/6, 1/4 or 0). Default: 1/6")
    ap.add_argument("-w", "--width", required=False, default=380, type=int,\
        help="Width of the input image (640 or 320). Default: 380")
    ap.add_argument("-r", "--ransacount", required=False, default=300, type=int,\
        help="Number of randomize samples for Ransac evaluation. Default: 300")
    ap.add_argument("-v", "--verification", required=False, type=str2bool, nargs='?',\
        const=True, default=False,\
        help="Activate the verification mode if set to True. Default: False")
    args = vars(ap.parse_args())

    # print(MTEAlgo[args["algo"]])
    # print(convert_to_float(args["crop"]))
    # print(args["width"])

    mte = MTE(mte_algo=MTEAlgo[args["algo"]], crop_margin=convert_to_float(args["crop"]), resize_width=args["width"], ransacount=args["ransacount"])
    if not args["verification"]:
        mte.listen_images()
    else:
        mte.check_reference()

