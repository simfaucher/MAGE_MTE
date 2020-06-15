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
from skimage.util import random_noise
import skimage
import imagezmq
import argparse
import csv
import shutil

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
DEBUG = True

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
        self.ref = None

        if self.mte_algo in (MTEAlgo.SIFT_KNN, MTEAlgo.SIFT_RANSAC):
            self.sift_engine = SIFTEngine(maxRansac=ransacount, width=self.resize_width, height=self.resize_height)
        else:
            self.vc_like_engine = VCLikeEngine()

    ### This function creates and initialize a writer
    ### In : name -> String being the name of the csv files that will be created, can be a path
    ### Out : Writer object pointing to name.csv
    def init_writer(self, name):
        result_csv = open(name+'.csv', 'w')
        metrics = ['Temps', 'Validité', 'Nombre de points interet', 'Nombre de match',
                   'Coefficient de translation', 'Coefficient de rotation',
                   'Distance D VisionCheck', 'Distance ROI 1',
                   'Distance ROI 2', 'Distance ROI 3', 'width=',
                   self.resize_width, 'height=', self.resize_height, 'type de flou',
                   'nb kp ref origine', 'nb kp ref reduit']
        writer = csv.DictWriter(result_csv, fieldnames=metrics)
        writer.writeheader()
        return writer

    ### This function fill a csv files with the data set as input
    ### In :    writer -> Object pointing to the csv file
    ###         results -> array containing the results of the recognition
    ###         blur -> string to associate a blur with a recognition
    ###         nb_kp_ref -> int for the number of keypoints in the potential reference at full size
    ###         nb_kp_ref_red -> int for the number of keypoints in the potential reference at redux size
    def fill_writer(self, writer, results, blur, nb_kp_ref, nb_kp_ref_red):
        if results["success"]:
            writer.writerow({'Temps' : results["timer"],
                             'Validité' : results["success"],
                             'Nombre de points interet': results["nb_kp"],
                             'Nombre de match' : results["nb_match"],
                             'Coefficient de translation' : results["sum_translation"],
                             'Coefficient de rotation' : results["sum_skew"],
                             'Distance D VisionCheck' : results["sum_distances"],
                             'Distance ROI 1' : results["dist_roi"][0],
                             'Distance ROI 2' : results["dist_roi"][1],
                             'Distance ROI 3' : results["dist_roi"][2],
                             'type de flou' : blur,
                             'nb kp ref origine' : nb_kp_ref,
                             'nb kp ref reduit' : nb_kp_ref_red})
        else:
            writer.writerow({'Temps' : results["timer"],
                             'Validité' : results["success"],
                             'Nombre de points interet': results["nb_kp"],
                             'Nombre de match' : results["nb_match"],
                             'Coefficient de translation' : results["sum_translation"],
                             'Coefficient de rotation' : results["sum_skew"],
                             'type de flou' : blur,
                             'nb kp ref origine' : nb_kp_ref,
                             'nb kp ref reduit' : nb_kp_ref_red})

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
    ### Out :   results -> array containing the results of the recognition
    def test_filter(self, blurred_image):
        start_frame_computing = time.time()
        dim = (self.resize_width, self.resize_height)
        gaussian_redux = cv2.resize(blurred_image, dim, interpolation=cv2.INTER_AREA)
        success, recog_ret_data, nb_kp, nb_match, sum_translation, \
            sum_skew, sum_distances, dist_roi, warped_img = self.recognition(-1, gaussian_redux)
        stop_frame_computing = time.time()
        results = {'success' : success,
                   'timer' : stop_frame_computing-start_frame_computing,
                   'recog_ret_data' : recog_ret_data,
                   'nb_kp' : nb_kp,
                   'nb_match' : nb_match,
                   'sum_translation' : sum_translation,
                   'sum_skew' : sum_skew,
                   'sum_distances' : sum_distances,
                   'dist_roi' : dist_roi,
                   'warped_img' : warped_img}
        return results

    ### Check if the images in a folder are valid for MTE
    ### In  :
    ### Out :
    def check_reference(self):
        kernel_size = 25
        sigma = 5
        kernel = 31

        kernel_v = np.zeros((kernel_size, kernel_size))
        kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
        kernel_v /= kernel_size

        kernel_h = np.zeros((kernel_size, kernel_size))
        kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
        kernel_h /= kernel_size

        writer_gaus = self.init_writer('gauss')
        writer_mvh = self.init_writer('horizontal')
        writer_mvv = self.init_writer('vertical')

        for file in os.listdir("videoForBenchmark/benchmark Validation capture/"):
            if not file.endswith(".jpg"):
                continue
            checkout = 0
            filename = "videoForBenchmark/benchmark Validation capture/"+file
            if DEBUG:
                print("Computing "+file)
            image_ref = cv2.imread(filename)
            try:
                os.mkdir(filename[:-4])
            except FileExistsError:
                shutil.rmtree(filename[:-4])
                os.mkdir(filename[:-4])
                if DEBUG:
                    print("Suppression du dossier existant")

            # Gaussian noise
            image_gaussian_blur = cv2.GaussianBlur(image_ref, (kernel, kernel), sigma)

            # Vertical motion blur.
            image_vertical_motion_blur = cv2.filter2D(image_ref, -1, kernel_v)

            # Horizontal motion blur.
            image_horizontal_motion_blur = cv2.filter2D(image_ref, -1, kernel_h)

            # Compute keypoints for reference
            image_ref_kp, _ = self.sift_engine.sift.detectAndCompute(image_ref, None)
            for i in range(len(image_ref_kp)):
                image_ref_kp[i].size = 2
            draw_kp_image_ref = cv2.drawKeypoints(image_ref, image_ref_kp, \
                np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite(filename[:-4]+"/image_ref.png", draw_kp_image_ref)

            # Resize reference and compute keypoints
            dim = (self.resize_width, self.resize_height)
            image_ref_reduite = cv2.resize(image_ref, dim, interpolation=cv2.INTER_AREA)
            image_ref_reduite_kp, _ = self.sift_engine.sift.detectAndCompute(image_ref_reduite, None)
            for i in range(len(image_ref_reduite_kp)):
                image_ref_reduite_kp[i].size = 1
            draw_kp_image_ref_reduite = cv2.drawKeypoints(image_ref_reduite, image_ref_reduite_kp, \
                np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite(filename[:-4]+"/image_ref_reduite.png", draw_kp_image_ref_reduite)
            lenght_kp_image_ref = len(image_ref_kp)
            lenght_kp_image_ref_red = len(image_ref_reduite_kp)

            # A writer for each image in his folder
            writer_for_image = self.init_writer(filename[:-4]+'/value')

            self.fake_init_for_reference(image_ref_reduite, image_ref)

            results = self.test_filter(image_gaussian_blur)
            if results["success"]:
                checkout += 1
                cv2.imwrite(filename[:-4]+"/homographieGaussiennek{}s{}.png".\
                    format(kernel, sigma), results["warped_img"])
            else:
                cv2.imwrite(filename[:-4]+"/echecGaussiennek{}s{}.png".\
                    format(kernel, sigma), results["warped_img"])
                # results["dist_roi"] = [0,0,0]
                results["sum_distances"] = 0
            self.fill_writer(writer_for_image, results, "gaussien", \
                lenght_kp_image_ref, lenght_kp_image_ref_red)
            self.fill_writer(writer_gaus, results, "gaussien", \
                lenght_kp_image_ref, lenght_kp_image_ref_red)

            ################# Flou vertical  #############################
            results = self.test_filter(image_vertical_motion_blur)

            if results["success"]:
                checkout += 1
                cv2.imwrite(filename[:-4]+"/homographieVerticale{}.png".\
                    format(kernel_size), results["warped_img"])
            else:
                # dist_roi=[0,0,0]
                cv2.imwrite(filename[:-4]+"/echecVerticale{}.png".\
                    format(kernel_size), results["warped_img"])
            self.fill_writer(writer_for_image, results, "vertical", \
                lenght_kp_image_ref, lenght_kp_image_ref_red)
            self.fill_writer(writer_mvv, results, "vertical", \
                lenght_kp_image_ref, lenght_kp_image_ref_red)
            ################# Flou horizontal  #############################
            results = self.test_filter(image_horizontal_motion_blur)

            if results["success"]:
                checkout += 1
                cv2.imwrite(filename[:-4]+"/homographieHorizontale{}.png".\
                    format(kernel_size), results["warped_img"])
            else:
                # dist_roi=[0,0,0]
                cv2.imwrite(filename[:-4]+"/echecHorizontale{}.png".\
                    format(kernel_size), results["warped_img"])
            self.fill_writer(writer_for_image, results, "horizontal", \
                lenght_kp_image_ref, lenght_kp_image_ref_red)
            self.fill_writer(writer_mvh, results, "horizontal", \
                lenght_kp_image_ref, lenght_kp_image_ref_red)

            if checkout == 3:
                self.save_reference_data(image_ref, image_ref_kp, \
                    image_ref_reduite, image_ref_reduite_kp)
                print(file + " est valide pour référence")
            else:
                print(file + " est invalide pour référence")

    def save_reference_data(self, image_full, kp_full, image_redux, kp_redux):
        dim = (640, 360)
        image_ref_half_redux = cv2.resize(image_full, dim, interpolation=cv2.INTER_AREA)
        image_ref_half_redux_kp, _ = self.sift_engine.sift.detectAndCompute(image_redux, None)
        for i in range(len(image_ref_half_redux_kp)):
            image_ref_half_redux_kp[i].size = 1
        ## TODO : save kp_full, image_ref_half_redux_kp and kp_redux

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
        sum(skews), sum_distances, distances, transformed

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
    mte.check_reference()
