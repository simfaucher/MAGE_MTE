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
VID=''
DETECTEUR='D2TierLowRes'
MATCH='Gpu'

# CAM_MATRIX = np.array([[954.16160543, 0., 635.29854945], \
#     [0., 951.09864051, 359.47108905],  \
#         [0., 0., 1.]])

VC_LIKE_ENGINE_MODE = False
SIFT_ENGINE_MODE = not VC_LIKE_ENGINE_MODE

class MTE:
    def __init__(self, mte_algo=MTEAlgo.SIFT_KNN, crop_margin=1.0/6, resize_width=640,ransacount=300):
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
        self.resize_width = resize_width
        self.resize_height = int((resize_width/16)*9)

        if self.mte_algo in (MTEAlgo.SIFT_KNN, MTEAlgo.SIFT_RANSAC):
            self.sift_engine = SIFTEngine(maxRansac = ransacount,width = self.resize_width,height = self.resize_height)
            if self.mte_algo == MTEAlgo.SIFT_KNN:
                VID = "SiftKnn"
            else:
                VID = "SiftRansac"
        elif self.mte_algo in (MTEAlgo.D2NET_KNN, MTEAlgo.D2NET_RANSAC):
            self.d2net_engine = D2NetEngine(max_edge=resize_width,max_sum_edges= resize_width + self.resize_height,\
                                            maxRansac = ransacount,width = self.resize_width,height = self.resize_height)
            if self.mte_algo == MTEAlgo.D2NET_KNN:
                VID = "D2netKnn"
            else:
                VID = "D2netRansac"
        else:
            self.vc_like_engine = VCLikeEngine()

        #csvWriter
        self.csvFile = open(VID+str(self.resize_width)+"x"+str(self.resize_height)+'.csv','w')

        metrics=['Temps','Nombre de points interet','Nombre de match',
                'Coefficient de translation','Coefficient de rotation',
                'Distance D VisionCheck','Distance ROI 1',
                'Distance ROI 2','Distance ROI 3','CropRef=False','width=',self.resize_width,'height=',self.resize_height]
        self.writer = csv.DictWriter(self.csvFile, fieldnames=metrics)
        self.writer.writeheader()

        #Behavior variables
        self.numberConsecutiveValidation = 0
        self.resolutionMax = (self.resize_width,self.resize_height)

    def listen_images(self):
        frameId = 0
        while True:  # show streamed images until Ctrl-C
            msg, image = self.image_hub.recv_image()
            #Fonction bloquante on peut donc lancer le timer juste aprèsc
            startFrameComputing = time.time()

            data = json.loads(msg)

            ret_data = {}

            if "error" in data and data["error"]:
                if CAPTURE_DEMO and self.out is not None:
                    print("No connection")
                    self.out.release()
                    self.out = None
                    cv2.destroyWindow("Matching result")
                continue

            imageForLearning = image
            dim = (self.resize_width, self.resize_height)
            image = cv2.resize(imageForLearning,dim, interpolation = cv2.INTER_AREA)

            mode = MTEMode(data["mode"])
            if mode == MTEMode.PRELEARNING:
                print("MODE prelearning")
                nb_kp = self.prelearning(image)
                # save_ref = "save_ref" in data and data["save_ref"]
                # ret_data["prelearning_pts"] = self.get_rectangle(0, image, force_new_ref=save_ref)
                ret_data["prelearning"] = {
                    "nb_kp": nb_kp
                }
                self.writer.writerow({'Temps' : time.time()-startFrameComputing ,
                                    'Nombre de points interet': nb_kp})
            elif mode == MTEMode.LEARNING:
                print("MODE learning")
                learning_id = self.learning(imageForLearning)

                ret_data["learning"] = {
                    "id": learning_id
                }
            elif mode == MTEMode.RECOGNITION:
                pov_id = data["pov_id"]
                # print("MODE recognition")
                success, recog_ret_data,nb_kp, nb_match, sumTranslation, sumSkew, sumD,distRoi,warpedImg = self.recognition(pov_id, image)
                stopFrameComputing = time.time()

                ret_data["recognition"] = recog_ret_data
                ret_data["recognition"]["success"] = success
                if success :
                    # cv2.imwrite("framing/homograhpeiFlou{}".format(frameId)+".png",warpedImg)
                    # cv2.imwrite("framing/resized{}".format(frameId)+".png",image)
                    # cv2.imwrite("framing/init{}".format(frameId)+".png",imageForLearning)
                    self.writer.writerow({'Temps' : stopFrameComputing-startFrameComputing ,
                                    'Nombre de points interet': nb_kp,
                                    'Nombre de match' : nb_match,
                                    'Coefficient de translation' : sumTranslation,
                                    'Coefficient de rotation' : sumSkew,
                                    'Distance D VisionCheck' : sumD,
                                    'Distance ROI 1' : distRoi[0],
                                    'Distance ROI 2' : distRoi[1],
                                    'Distance ROI 3' : distRoi[2]})
                else :
                    self.writer.writerow({'Temps' : stopFrameComputing-startFrameComputing ,
                                    'Nombre de points interet': nb_kp,
                                    'Nombre de match' : nb_match,
                                    'Coefficient de translation' : sumTranslation,
                                    'Coefficient de rotation' : sumSkew})
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

            frameId = frameId + 1

    #This function return a code that will inform the client what to do
    # TODO : create dommain/behavior.py
    def behavior(self,warped,shapeValue,sketchValue,colorValue):
        if warped.shape[0] < 780:
            thresholdShape = 2500
            thresholdSketch = 870
            thresholdColor = 9580
            errorFactor=1.5
        else:
            thresholdShape = 2500
            thresholdSketch = 870
            thresholdColor = 9580
            errorFactor=2

        #Used to reduce dimension
        reduceShape = shapeValue < thresholdShape * errorFactor
        reduceSketch = sketchValue < thresholdSketch* errorFactor
        reduceColor = colorValue < thresholdColor * errorFactor

        #Used to maintain actual dimension
        correctShape = shapeValue < thresholdShape
        correctSketch = sketchValue < thresholdSketch
        correctColor = colorValue < thresholdColor

        if reduceShape+reduceSketch+reduceColor >= 2:
            self.numberConsecutiveValidation = 0
            return "Reduction"
        if correctShape+correctSketch+correctColor = 3:
            self.numberConsecutiveValidation += 1
        elif correctShape+correctSketch+correctColor >= 1:
            self.numberConsecutiveValidation -= 1
        else
            if warped.shape[:2] = self.resolutionMax:
                return "perte cible"
            else:
                self.numberConsecutiveValidation -= 1
                return "Augmentation"

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
        # Enregistrement de l'image de référence en 640 pour SIFT + VC léger et 4K pour VCE
        learning_id = self.repo.save_new_pov(full_image)

        success, learning_data = self.repo.get_pov_by_id(learning_id, resize_width=self.resize_width)
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
        sum_distances = 9999
        distances = 9999
        learning_data = self.get_learning_data(pov_id)

        fps = FPS().start()
        if self.mte_algo == MTEAlgo.VC_LIKE:
            success, scale, skew, transformed = self.vc_like_engine.find_target(image, learning_data)
            # cv2.imshow("VC-like engine", transformed)
        elif self.mte_algo in (MTEAlgo.SIFT_KNN, MTEAlgo.SIFT_RANSAC):
            success, scale, skew, translation, transformed,nb_match, nb_kp,sumTranslation, sumSkew = self.sift_engine.recognition(image, learning_data,self.mte_algo)
        else:
            success, scale, skew, translation, transformed,nb_match, nb_kp,sumTranslation, sumSkew = self.d2net_engine.recognition(image, learning_data,self.mte_algo)

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

        cv2.imshow("Transformed", transformed)
        cv2.waitKey(1)

        ret_data["success"] = success and ml_success

        return success, ret_data,nb_kp, nb_match, sumTranslation,\
        sumSkew, sum_distances,distances,transformed

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
                success, learning_data = self.repo.get_pov_by_id(pov_id, resize_width=self.resize_width)
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
    ap.add_argument("-w", "--width", required=False, default=640, type=int,\
        help="Width of the input image (640 or 320). Default: 640")
    ap.add_argument("-r", "--ransacount", required=False, default=300, type=int,\
        help="Number of randomize samples for Ransac evaluation. Default: 300")
    args = vars(ap.parse_args())

    print(MTEAlgo[args["algo"]])
    print(convert_to_float(args["crop"]))
    print(args["width"])

    mte = MTE(mte_algo=MTEAlgo[args["algo"]], crop_margin=convert_to_float(args["crop"]), resize_width=args["width"],ransacount=args["ransacount"])
    mte.listen_images()
