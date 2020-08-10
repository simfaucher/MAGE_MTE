"""
    Server side to catch a camera stream from a client
"""

import os
import math

import argparse
import csv
import json
from datetime import datetime
import cv2
import numpy as np


from imutils.video import FPS
import imagezmq

from Domain.MTEMode import MTEMode
from Domain.ErrorLearning import ErrorLearning
from Domain.ErrorRecognition import ErrorRecognition
from Domain.ErrorInitialize import ErrorInitialize
from Domain.MTEAlgo import MTEAlgo
from Domain.LearningData import LearningData
from Domain.MTEResponse import MTEResponse
from Domain.MTEThreshold import MTEThreshold
from Domain.RecognitionData import RecognitionData
from Domain.ResponseData import ResponseData
from Domain.UserInformation import UserInformation
from Repository import Repository

from MLValidation import MLValidation
from SIFTEngine import SIFTEngine
from D2NetEngine import D2NetEngine
from VCLikeEngine import VCLikeEngine

# CAM_MATRIX = np.array([[954.16160543, 0., 635.29854945], \
#     [0., 951.09864051, 359.47108905],  \
#         [0., 0., 1.]])

MIN_VALIDATION_COUNT = 5

class MTE:
    """
    This class initializes a server that will listen to client
    and will compute motion tracking for the client.
    """

    def __init__(self, mte_algo=MTEAlgo.SIFT_KNN, crop_margin=1.0/6,\
         resize_width=640, ransacount=300):
        print("Launching server")
        self.image_hub = imagezmq.ImageHub()
        self.image_hub.zmq_socket.RCVTIMEO = 3000
        # self.image_hub = imagezmq.ImageHub(open_port='tcp://192.168.43.39:5555')

        self.repo = Repository()

        self.learning_db = []
        self.last_learning_data = None

        # ML validation
        self.ml_validator = MLValidation()

        self.format_resolution = None
        self.width_small = None
        self.width_medium = None
        self.width_large = None

        # Motion tracking engines
        self.mte_algo = mte_algo
        self.crop_margin = crop_margin
        self.resize_width = resize_width
        self.resize_height = int(resize_width*(1/(16/9)))
        self.validation_width = None
        self.validation_height = None
        if self.mte_algo in (MTEAlgo.D2NET_KNN, MTEAlgo.D2NET_RANSAC):
            self.d2net_engine = D2NetEngine(max_edge=resize_width, \
                                            max_sum_edges=resize_width + self.resize_height,\
                                            maxRansac=ransacount, width=self.resize_width, \
                                            height=self.resize_height)
        elif self.mte_algo == MTEAlgo.VC_LIKE:
            self.vc_like_engine = VCLikeEngine()
        else:
            self.sift_engine = SIFTEngine(maxRansac=ransacount)

        self.threshold_small = MTEThreshold(100, 45, 3500, 1100, 12000, 4000, 13000)
        self.threshold_medium = MTEThreshold(100, 70, 3400, 1200, 14000, 5000, 18000)
        self.threshold_large = MTEThreshold(3500, 180, 3100, 750, 13000, 5500, 20000)

        self.rollback = 0
        self.validation = 0
        self.devicetype = "CPU"
        self.resolution_change_allowed = 3

        self.reference = LearningData()

        self.debug = None
        self.result_csv = None

    def init_log(self, name):
        """ This function creates and initializes a writer.

        In : name -> String being the name of the csv file that will be created, can be a path
        Out : Writer object pointing to name.csv
        """
        self.result_csv = open(name+'.csv', 'w')
        metrics = ['Success', 'Number of keypoints', 'Number of matches',
                   'Distance Kirsh', 'Distance Canny', 'Distance Color',
                   'Translation x', 'Translation y',
                   'Scale x', 'Scale y',
                   'Flag', 'Code', 'Direction',
                   'Blurred']
        writer = csv.DictWriter(self.result_csv, fieldnames=metrics)
        writer.writeheader()
        return writer

    def fill_log(self, writer, recognition, response, is_blurred):
        """ This function fill a csv files with the data set as input.

        In :    writer -> Object pointing to the csv file
                recognition -> results of the recognition
                response -> data that will be sent to client
                is blurred -> is the current image blurred
        """
        if not isinstance(recognition.dist_roi, int):
            distance_kirsh = recognition.dist_roi[0]
            distance_canny = recognition.dist_roi[1]
            distance_color = recognition.dist_roi[2]
        else:
            distance_kirsh = ""
            distance_canny = ""
            distance_color = ""

        writer.writerow({'Success' : recognition.success,
                         'Number of keypoints' : recognition.nb_kp,
                         'Number of matches': recognition.nb_match,
                         'Distance Kirsh' : distance_kirsh,
                         'Distance Canny' : distance_canny,
                         'Distance Color' : distance_color,
                         'Translation x' : recognition.translations[0],
                         'Translation y' : recognition.translations[1],
                         'Scale x' : recognition.scales[0],
                         'Scale y' : recognition.scales[1],
                         'Flag' : response.flag,
                         'Code' : response.status,
                         'Direction' : response.user_information,
                         'Blurred' : is_blurred})

    def set_mte_parameters(self, ratio):
        """Edit values for globals parameters of the motion tracking engine."""

        if math.isclose(ratio, 16/9, rel_tol=1e-5):
            self.width_small = 400
            self.width_medium = 660
            self.width_large = 1730
            self.format_resolution = 16/9
        elif math.isclose(ratio, 4/3, rel_tol=1e-5):
            self.width_small = 350
            self.width_medium = 570
            self.width_large = 1730
            self.format_resolution = 4/3
        else:
            print("What kind of format is that ?")
            return False

        self.validation_width = self.width_small
        self.validation_height = int(self.validation_width*(1/self.format_resolution))
        if self.mte_algo in (MTEAlgo.SIFT_KNN, MTEAlgo.SIFT_RANSAC):
            self.sift_engine.set_parameters(self.width_small, self.width_medium,\
                                            self.width_large, self.format_resolution)
        elif self.mte_algo == MTEAlgo.VC_LIKE:
            self.vc_like_engine.set_parameters(self.format_resolution)

        return True

    def listen_images(self):
        """Receive a frame and an action from client then compute required operation

        The behaviour depend of the mode send : PRELEARNING/LEARNING/RECOGNITION
        This function has no proper value to return but will send a message to the client
        containing the operations' results.
        """

        while True:  # show streamed images until Ctrl-C
            msg, image = self.image_hub.recv_image()

            data = json.loads(msg)

            if "error" in data and data["error"]:
                continue

            if MTEMode(data["mode"]) == MTEMode.VALIDATION_REFERENCE:
                self.rollback = 0
                self.validation = 0
                self.resolution_change_allowed = 3
                resolution_valid = self.set_mte_parameters(image.shape[1]/image.shape[0])
                if resolution_valid:
                    status = self.learning(image)
                    to_send = {
                        "status": status.value,
                        "mte_parameters": {}
                    }
                    # self.reference.mte_parameters["ratio"] = self.format_resolution
                    if status == ErrorLearning.SUCCESS:
                        to_send["mte_parameters"] = self.reference.change_parameters_type_for_sending()
                else:
                    to_send = {
                        "status" : ErrorLearning.INVALID_FORMAT.value
                    }

            elif MTEMode(data["mode"]) == MTEMode.INITIALIZE_MTE:
                if data["id_ref"] is None:
                    to_send = {
                        "status" : ErrorInitialize.ERROR.value
                    }
                else:
                    self.format_resolution = data["mte_parameters"]["ratio"]
                    self.set_mte_parameters(self.format_resolution)
                    init_status = self.reference.initialiaze_control_assist\
                        (data["id_ref"], data["mte_parameters"])
                    
                    if self.mte_algo == MTEAlgo.VC_LIKE:
                        self.vc_like_engine.init_engine(self.reference)

                    if init_status == 0:
                        log_location = os.path.join("logs", "ref"+str(self.reference.id_ref))
                        if not os.path.exists(log_location):
                            os.makedirs(log_location)
                        log_name = datetime.now().strftime("%m%d%Y_%H%M%S")
                        log_path = os.path.join(log_location, log_name)
                        log_writer = self.init_log(log_path)
                    to_send = {
                        "status" : init_status
                    }

            elif MTEMode(data["mode"]) == MTEMode.MOTION_TRACKING:
                if self.reference.id_ref is None:
                    to_send = {
                        "status" : ErrorRecognition.ENGINE_IS_NOT_INITIALIZED.value
                    }
                elif data["id_ref"] != self.reference.id_ref:
                    print("Wrong initialization.")
                    to_send = {
                        "status" : ErrorRecognition.MISMATCH_REF.value
                    }
                else:
                    self.debug = image
                    if self.devicetype == "CPU" and image.shape[1] > self.width_medium:
                        image = cv2.resize(image, (self.width_medium, \
                            int(self.width_medium*(1/self.format_resolution))),\
                                 interpolation=cv2.INTER_AREA)
                    results = RecognitionData(*self.recognition(image))
                    if image.shape[1] == self.width_small:
                        response = self.behaviour_width_small(results)
                    elif image.shape[1] == self.width_medium:
                        response = self.behaviour_width_medium(results)
                    elif image.shape[1] == self.width_large:
                        response = self.behaviour_width_large(results)
                    else:
                        print("Image size not supported.")
                        response = ResponseData(\
                                                [self.width_small,\
                                                self.width_small*self.format_resolution],\
                                                MTEResponse.RED, 0, 0, UserInformation.CENTERED, \
                                                0, 0, ErrorRecognition.MISMATCH_SIZE_WITH_REF)

                    # If we can capture
                    is_blurred = False
                    if self.validation > MIN_VALIDATION_COUNT:
                        self.validation = MIN_VALIDATION_COUNT
                    if self.validation == MIN_VALIDATION_COUNT:
                        is_blurred = self.is_image_blurred(image, \
                            size=int(response.requested_image_size[0]/18), thresh=10)
                        # if the image is not blurred else we just return green
                        if not is_blurred[1] and response.user_information == UserInformation.CENTERED:
                            response.flag = MTEResponse.CAPTURE
                    to_send = response.to_dict()
                    self.fill_log(log_writer, results, response, is_blurred)

            elif MTEMode(data["mode"]) == MTEMode.CLEAR_MTE:
                to_send = {
                    "status" : self.reference.clean_control_assist(data["id_ref"])
                }
                self.result_csv = self.result_csv.close()
            else:
                # Impossible
                to_send = {
                    "status" : 1
                }
                print("{} is an unknown mode.".format(data["mode"]))

            self.image_hub.send_reply(json.dumps(to_send).encode())

    def is_image_blurred(self, image, size=60, thresh=15):
        """Check if an image is blurred. Return a tuple (mean: float, blurred: bool)

        Keyword arguments:
        image -> the image to test as array
        size -> the radius size around the center that will be used in FFTShift (default 60)
        thresh -> the threshold value for the magnitude comparaison (default 15)
        """

        (height, width, _) = image.shape
        (center_x, center_y) = (int(width / 2.0), int(height / 2.0))
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)
        fft_shift[center_y - size:center_y + size, center_x - size:center_x + size] = 0
        fft_shift = np.fft.ifftshift(fft_shift)
        recon = np.fft.ifft2(fft_shift)
        magnitude = 20 * np.log(np.abs(recon))
        mean = np.mean(magnitude)
        return (mean, mean <= thresh)

    def compute_direction(self, translation_value, scale_value, size_w):
        """Return a string representing a cardinal direction.

        Keyword arguments:
        translation -> tuple containing homographic estimations of x,y
        size_w -> the width of the current image
        """

        divider = 20
        center = (translation_value[0]*scale_value[0]+size_w/3, \
            translation_value[1]*scale_value[1]+int((size_w*(1/self.format_resolution))/3))

        direction = UserInformation.CENTERED

        size_h = int(size_w*(1/self.format_resolution))
        if center[1] < (size_h/2 - size_h/divider):
            if center[0] < (size_w/2 - size_w/divider):
                direction = UserInformation.UP_LEFT
            elif center[0] > (size_w/2 + size_w/divider):
                direction = UserInformation.UP_RIGHT
            else:
                direction = UserInformation.UP

        elif center[1] > (size_h/2 + size_h/divider):
            if center[0] < (size_w/2 - size_w/divider):
                direction = UserInformation.DOWN_LEFT
            elif center[0] > (size_w/2 + size_w/divider):
                direction = UserInformation.DOWN_RIGHT
            else:
                direction = UserInformation.DOWN

        else:
            if center[0] < (size_w/2 - size_w/divider):
                direction = UserInformation.LEFT
            elif center[0] > (size_w/2 + size_w/divider):
                direction = UserInformation.RIGHT
            else:
                direction = UserInformation.CENTERED

        # center_kp = cv2.KeyPoint(center[0], center[1], 8)
        # to_draw = cv2.drawKeypoints(self.debug, [center_kp], \
        # np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.putText(to_draw, "Direction: "+direction.value, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
        #                     (255, 0, 0), 2)
        # cv2.imshow("Direction", to_draw)
        # cv2.waitKey(1)
        return direction

    def red_width_small(self):
        """Critical recognition behaviour for an image (_, 380).
        Return a ResponseData and change global variables.
        """

        width = self.width_small
        self.rollback += 1
        if self.validation > 0:
            self.validation -= 1
        if self.rollback >= 5:
            self.rollback = 0
            width = self.width_medium
        return ResponseData([width, int(width*(1/self.format_resolution))],\
             MTEResponse.RED, None, None, None, None, None)

    def orange_behaviour(self, results, width):
        """Uncertain recognition behaviour.
        Return a ResponseData and change global variables.

        Keyword arguments:
        results -> the RecognitionData
        size -> the width of the image
        """

        if self.validation > 0:
            self.validation -= 1
        if width == self.width_medium and self.rollback > 0:
            self.rollback -= 1

        return ResponseData([width, int(width*(1/self.format_resolution))], MTEResponse.ORANGE,\
                            results.translations[0], results.translations[1], \
                            self.compute_direction(results.translations, results.scales, width), \
                            results.scales[0], results.scales[1])

    def red_width_medium(self):
        """Critical recognition behaviour for an image (_, 640).
        Return a ResponseData and change global variables.
        """

        width = self.width_medium
        msg = MTEResponse.RED
        if self.devicetype == "CPU":
            self.validation = 0
            msg = MTEResponse.TARGET_LOST
        else:
            self.rollback += 1
            if self.rollback >= 5:
                width = self.width_large
                self.rollback = 0
        return ResponseData([width, int(width*(1/self.format_resolution))],\
             msg, None, None, None, None, None)

    def green_width_medium(self, results):
        """Behaviour for an image (_, 640) when the flag turns on to be green.
        Return a ResponseData and change global variable.

        Keyword arguments:
        results -> the RecognitionData
        """

        width = self.width_medium
        if self.resolution_change_allowed > 0:
            self.resolution_change_allowed -= 1
            width = self.width_small
            self.validation = 0
            self.rollback = 0
        else:
            self.validation += 1
        return ResponseData(\
            [width, int(width*(1/self.format_resolution))],\
            MTEResponse.GREEN,\
            results.translations[0], results.translations[1],\
            self.compute_direction(results.translations,\
                results.scales, self.width_medium), \
            results.scales[0], results.scales[1])

    def lost_width_large(self):
        """Behaviour for a image (_, 1730) when the target is lost
        Return a ResponseData.
        """

        msg = MTEResponse.TARGET_LOST
        return ResponseData([self.width_large, int(self.width_large*(1/self.format_resolution))],\
             msg, None, None, None, None, None)

    def behaviour_width_small(self, results):
        """Global behaviour for recognition of image (_,380).
        Based on the activity diagram.
        Return a ResponseData.

        Keyword arguments:
        results -> the RecognitionData
        """

        response = MTEResponse.RED
        # If not enough keypoints
        if results.nb_kp < self.threshold_small.nb_kp:
            response_for_client = self.red_width_small()
            response_for_client.set_status(ErrorRecognition.NOT_ENOUGHT_KEYPOINTS)
            print("Not enought keypoints")
        # If not enough matches
        elif results.nb_match < self.threshold_small.nb_match:
            # If homography doesn't even start
            if results.nb_match < 30:
                response_for_client = self.red_width_small()
                response_for_client.set_status(ErrorRecognition.NOT_ENOUGHT_MATCH_CRITICAL)
            else:
                response_for_client = self.orange_behaviour(results, self.width_small)
                response_for_client.set_status(ErrorRecognition.NOT_ENOUGHT_MATCH)
        else:
            response = MTEResponse.GREEN
            # If not centered with target
            if not results.success:
                self.validation = 0
                response_for_client = ResponseData(\
                                [self.width_small,\
                                int(self.width_small*(1/self.format_resolution))],\
                                response,\
                                results.translations[0], results.translations[1], \
                                self.compute_direction(results.translations,\
                                    results.scales, self.width_small), \
                                results.scales[0], results.scales[1],\
                                status=ErrorRecognition.NOT_CENTERED_WITH_TARGET)
            else:
                dist_kirsh = results.dist_roi[0] < self.threshold_small.mean_kirsh
                dist_canny = results.dist_roi[1] < self.threshold_small.mean_canny
                dist_color = results.dist_roi[2] < self.threshold_small.mean_color
                # If 0 or 1 mean valid
                if int(dist_kirsh)+int(dist_canny)+int(dist_color) < 2:
                    response_for_client = self.orange_behaviour(results, self.width_small)
                    response_for_client.set_status(ErrorRecognition.MEANS_OUT_OF_LIMITS)
                else:
                    dist_kirsh = results.dist_roi[0] < self.threshold_small.kirsh_aberration
                    dist_color = results.dist_roi[2] < self.threshold_small.color_aberration
                    # If no aberration
                    if int(dist_kirsh)+int(dist_color) == 2:
                        self.validation += 1
                        self.rollback = 0
                        status = ErrorRecognition.SUCCESS
                    else:
                        response = MTEResponse.ORANGE
                        status = ErrorRecognition.ABERRATION_VALUE
                    response_for_client = ResponseData(\
                        [self.width_small, int(self.width_small*(1/self.format_resolution))],\
                        response,\
                        results.translations[0], results.translations[1], \
                        self.compute_direction(results.translations, \
                            results.scales, self.width_small), \
                        results.scales[0], results.scales[1], status=status)

        if response_for_client.flag == MTEResponse.GREEN:
            self.rollback = 0
        return response_for_client

    def behaviour_width_medium(self, results):
        """Global behaviour for recognition of image (_,640).
        Based on the activity diagram.
        Return a ResponseData.

        Keyword arguments:
        results -> the RecognitionData
        """

        if results.nb_kp < self.threshold_medium.nb_kp:
            response_for_client = self.red_width_medium()
            response_for_client.set_status(ErrorRecognition.NOT_ENOUGHT_KEYPOINTS)
        # If not enough matches
        elif results.nb_match < self.threshold_medium.nb_match:
            # If homography doesn't even start
            if results.nb_match < 30:
                response_for_client = self.red_width_medium()
                response_for_client.set_status(ErrorRecognition.NOT_ENOUGHT_MATCH_CRITICAL)
            else:
                response_for_client = self.orange_behaviour(results, self.width_medium)
                response_for_client.set_status(ErrorRecognition.NOT_ENOUGHT_MATCH)
        else:
            response = MTEResponse.GREEN
            # If not centered with target
            if not results.success:
                response_for_client = ResponseData([self.width_medium,\
                    int(self.width_medium*(1/self.format_resolution))], response,\
                    results.translations[0], results.translations[1], \
                    self.compute_direction(results.translations,\
                        results.scales, self.width_medium), \
                    results.scales[0], results.scales[1],\
                    status=ErrorRecognition.NOT_CENTERED_WITH_TARGET)
            else:
                dist_kirsh = results.dist_roi[0] < self.threshold_medium.mean_kirsh
                dist_canny = results.dist_roi[1] < self.threshold_medium.mean_canny
                dist_color = results.dist_roi[2] < self.threshold_medium.mean_color
                # If 0 or 1 mean valid
                if int(dist_kirsh)+int(dist_canny)+int(dist_color) < 2:
                    response_for_client = self.orange_behaviour(results, self.width_medium)
                    response_for_client.set_status(ErrorRecognition.MEANS_OUT_OF_LIMITS)
                # If all means are valids
                elif int(dist_kirsh)+int(dist_canny)+int(dist_color) == 3:
                    response_for_client = self.green_width_medium(results)
                    response_for_client.set_status(ErrorRecognition.SUCCESS)
                else:
                    dist_kirsh = results.dist_roi[0] < self.threshold_medium.kirsh_aberration
                    dist_color = results.dist_roi[2] < self.threshold_medium.color_aberration
                    # If no aberration
                    if int(dist_kirsh)+int(dist_color) == 2:
                        response_for_client = self.green_width_medium(results)
                        response_for_client.set_status(ErrorRecognition.SUCCESS)
                    else:
                        response_for_client = self.orange_behaviour(results, self.width_medium)
                        response_for_client.set_status(ErrorRecognition.ABERRATION_VALUE)

        if response_for_client.flag == MTEResponse.GREEN:
            self.rollback = 0
        return response_for_client

    def behaviour_width_large(self, results):
        """Global behaviour for recognition of image (_,1730).
        Based on the activity diagram.
        Return a ResponseData.

        Keyword arguments:
        results -> the RecognitionData
        """

        if results.nb_kp < self.threshold_large.nb_kp:
            response_for_client = self.lost_width_large()
            response_for_client.set_status(ErrorRecognition.NOT_ENOUGHT_KEYPOINTS)
        # If not enough matches
        elif results.nb_match < self.threshold_large.nb_match:
            response_for_client = self.lost_width_large()
            response_for_client.set_status(ErrorRecognition.NOT_ENOUGHT_MATCH_CRITICAL)
        else:
            response = MTEResponse.GREEN
            # If not centered with target
            if not results.success:
                response_for_client = ResponseData([self.width_large,\
                    int(self.width_large*(1/self.format_resolution))], response,\
                    results.translations[0], results.translations[1], \
                    self.compute_direction(results.translations,\
                        results.scales, self.width_large), \
                    results.scales[0], results.scales[1],\
                    status=ErrorRecognition.SUCCESS)
            else:
                dist_kirsh = results.dist_roi[0] < self.threshold_large.mean_kirsh
                dist_canny = results.dist_roi[1] < self.threshold_large.mean_canny
                dist_color = results.dist_roi[2] < self.threshold_large.mean_color
                # If 0 or 1 mean valid
                if dist_kirsh+dist_canny+dist_color < 2:
                    response_for_client = self.lost_width_large()
                    response_for_client.set_status(ErrorRecognition.MEANS_OUT_OF_LIMITS)
                # If all means are valids
                elif dist_kirsh+dist_canny+dist_color == 3:
                    response_for_client = ResponseData([self.width_medium,\
                        int(self.width_medium*(1/self.format_resolution))], response,\
                        results.translations[0], results.translations[1], \
                        self.compute_direction(results.translations,\
                            results.scales, self.width_large), \
                        results.scales[0], results.scales[1],\
                        status=ErrorRecognition.SUCCESS)
                else:
                    dist_kirsh = results.dist_roi[0] < self.threshold_large.kirsh_aberration
                    dist_color = results.dist_roi[2] < self.threshold_large.color_aberration
                    # If no aberration
                    if dist_kirsh+dist_color == 2:
                        size = self.width_medium
                        status = ErrorRecognition.SUCCESS
                    else:
                        response = MTEResponse.ORANGE
                        size = self.width_large
                        status = ErrorRecognition.ABERRATION_VALUE
                    response_for_client = ResponseData([size,\
                        int(size*(1/self.format_resolution))], response,\
                        results.translations[0], results.translations[1], \
                        self.compute_direction(results.translations,\
                            results.scales, self.width_large), \
                        results.scales[0], results.scales[1], status=status)
        return response_for_client

    def fake_init_for_reference(self, image_ref):
        """Initialize learning datas with the reference and avoid the use of database.

        Keyword arguments:
        image_ref -> int array of the reference in full size
        """

        self.reference = LearningData()

        if self.mte_algo in (MTEAlgo.D2NET_KNN, MTEAlgo.D2NET_RANSAC):
            self.d2net_engine.learn(self.reference, crop_image=True, crop_margin=self.crop_margin)
        elif self.mte_algo == MTEAlgo.VC_LIKE:
            self.vc_like_engine.learn(image_ref, self.reference)
            self.vc_like_engine.init_engine(self.reference)
        else:
            self.sift_engine.learn(image_ref, self.reference, \
                crop_image=True, crop_margin=self.crop_margin)
        self.ml_validator.learn(self.reference, image_ref)
        self.last_learning_data = self.reference

    def test_filter(self, blurred_image):
        """Test the recognition between the input and the image learned with fakeInitForReference.
        Return a RecognitionData

        Keyword arguments:
        blurred_image -> int array of the blurred reference
        """

        dim = (self.validation_width, self.validation_height)
        blurred_redux = cv2.resize(blurred_image, dim, interpolation=cv2.INTER_AREA)
        results = RecognitionData(*self.recognition(blurred_redux))

        return results

    def check_reference(self, image_ref):
        """Check if the image given is a valid reference.
        Return a dictinnary with 2 boolean:
        success -> is the given image valid as reference
        blurred -> is the given image blurred

        Keyword arguments:
        image_ref -> int array of the potential reference
        """

        size = int(image_ref.shape[1]/18)
        blurred = self.is_image_blurred(image_ref, \
                        size=size, thresh=10)
        if blurred[1]:
            print("The image is blurred")
            return ErrorLearning.ERROR_REFERENCE_IS_BLURRED

        # kernel_size = 10
        # sigma = 3
        # kernel = 15
        kernel_size = 2
        sigma = 3
        kernel = 1

        kernel_v = np.zeros((kernel_size, kernel_size))
        kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
        kernel_v /= kernel_size

        kernel_h = np.zeros((kernel_size, kernel_size))
        kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
        kernel_h /= kernel_size

        self.fake_init_for_reference(image_ref)

        # Gaussian noise
        image_gaussian_blur = cv2.GaussianBlur(image_ref, (kernel, kernel), sigma)
        results = self.test_filter(image_gaussian_blur)
        if not results.success:
            print("Failure gaussian blur")
            return ErrorLearning.GAUSSIAN_BLUR_FAILURE

        # Vertical motion blur.
        image_vertical_motion_blur = cv2.filter2D(image_ref, -1, kernel_v)
        results = self.test_filter(image_vertical_motion_blur)

        if not results.success:
            print("Failure vertical blur")
            return ErrorLearning.VERTICAL_BLUR_FAILURE

        # Horizontal motion blur.
        image_horizontal_motion_blur = cv2.filter2D(image_ref, -1, kernel_h)
        results = self.test_filter(image_horizontal_motion_blur)

        if not results.success:
            print("Failure horizontal blur")
            return ErrorLearning.HORIZONTAL_BLUR_FAILURE

        # All 3 noises are valid
        print("Valid for reference.")

        return ErrorLearning.SUCCESS

    def learning(self, full_image):
        """Test if the given image can be used as reference.
        Return a dictionary containing :
        success for global success
        blurred to indicate if the image is blurred
        learning_id for the position of the image in the DB

        Keyword arguments:
        full_image -> int array of the image
        """

        return self.check_reference(full_image)

    def recognition(self, image):
        """Compute the homography using the running engine.
        Computation is made between the image from the video stream and
        the image that have been learn with its id.

        Keyword arguments:
        pov_id -> int for the position of the reference in the DB
        image -> int array for the video stream
        """

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

        fps = FPS().start()
        nb_matches = 0
        if self.mte_algo == MTEAlgo.VC_LIKE:
            nb_kp = 300
            nb_matches = 150
            success, scales, skews, translation, transformed = self.vc_like_engine.\
                find_target(image, self.reference)
            # cv2.imshow("VC-like engine", transformed)
        elif self.mte_algo in (MTEAlgo.D2NET_KNN, MTEAlgo.D2NET_RANSAC):
            success, scales, skews, translation, transformed, nb_matches, \
                nb_kp = self.d2net_engine.recognition(image, self.reference, self.mte_algo)
        else:
            success, scales, skews, translation, transformed, nb_matches, \
                nb_kp = self.sift_engine.recognition(image, self.reference, self.mte_algo)

        scale_x, scale_y = scales
        skew_x, skew_y = skews
        scale = max(scale_x, scale_y)
        skew = max(skew_x, skew_y)

        # ML validation
        ml_success = False
        if success:
            ml_success, sum_distances, distances = self.ml_validator.\
                validate(self.reference, transformed)

        fps.update()
        fps.stop()

        if not ml_success:
            # Scale
            if scale < SIFTEngine.HOMOGRAPHY_MIN_SCALE:
                ret_data["scale"] = "far"
            elif scale > SIFTEngine.HOMOGRAPHY_MAX_SCALE:
                ret_data["scale"] = "close"

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

        # cv2.imshow("Transformed", transformed)
        # cv2.waitKey(1)

        ret_data["success"] = success and ml_success

        return success, ret_data, nb_kp, nb_matches, sum(translation),\
        sum(skews), sum_distances, distances, transformed, scales, translation

def str2bool(string_value):
    """Convert a string to a bool."""

    if isinstance(string_value, bool):
        return string_value
    if string_value.lower() in ('yes', 'true', 't', 'y', '1', 'oui', 'o', 'vrai', 'v'):
        return True
    elif string_value.lower() in ('no', 'false', 'f', 'n', '0', 'non', 'faux'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    def convert_to_float(frac_str):
        """Convert a fraction written as string to a float"""

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
    ap.add_argument("-a", "--algo", required=False, default="VC_LIKE",\
        help="Feature detection algorithm (SIFT_KNN or SIFT_RANSAC). Default: SIFT_KNN")
    ap.add_argument("-c", "--crop", required=False, default="1/6",\
        help="Part to crop around the center of the image (1/6, 1/4 or 0). Default: 1/6")
    ap.add_argument("-w", "--width", required=False, default=380, type=int,\
        help="Width of the input image (640 or 320). Default: 380")
    ap.add_argument("-r", "--ransacount", required=False, default=300, type=int,\
        help="Number of randomize samples for Ransac evaluation. Default: 300")
    # ap.add_argument("-v", "--verification", required=False, type=str2bool, nargs='?',\
    #     const=True, default=False,\
    #     help="Activate the verification mode if set to True. Default: False")
    args = vars(ap.parse_args())

    mte = MTE(mte_algo=MTEAlgo[args["algo"]], crop_margin=convert_to_float(args["crop"]),\
         resize_width=args["width"], ransacount=args["ransacount"])
    mte.listen_images()
