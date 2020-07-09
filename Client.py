"""
    Client side to stream a camera to a server
"""

import os
import sys
import time
import json
import numpy as np
import cv2
import imutils
from imutils.video import FPS
import imagezmq

from Domain.ErrorLearning import ErrorLearning
from Domain.ErrorRecognition import ErrorRecognition
from Domain.LearningData import LearningData
from Domain.MTEMode import MTEMode


CAPTURE_DEMO = False
DEMO_FOLDER = "demo/"

MODE_CAMERA = False
MODE_VIDEO = not MODE_CAMERA

# T1.1
# VIDEO_PATH = "videos/T1.1/VID_20200302_144048.mp4"
# LEARNING_IMAGE_PATH = "videos/T1.1/vlcsnap-2020-03-02-15h59m47s327.png"

# T1.2
# VIDEO_PATH = "videos/T1.2/Zoom/VID_20200302_144327.mp4"
# LEARNING_IMAGE_PATH = "videos/T1.2/Zoom/vlcsnap-2020-03-02-16h00m31s968.png"

# T1.3
# VIDEO_PATH = "videos/T1.3/Zoom/VID_20200302_144507.mp4"
# LEARNING_IMAGE_PATH = "videos/T1.3/Zoom/vlcsnap-2020-03-02-16h01m23s741.png"

# T1.4
# VIDEO_PATH = "videos/T1.4/VID_20200302_144814.mp4"
# LEARNING_IMAGE_PATH = "videos/T1.4/vlcsnap-2020-03-02-16h02m56s976.png"
# LEARNING_IMAGE_PATH = "videos/T1.4/vlcsnap-2020-03-02-16h02m33s403.png"

# T2.1
# VIDEO_PATH = "videos/T2.1/T2.1-rotated.mp4"
# LEARNING_IMAGE_PATH = "videos/T2.1/vlcsnap-2020-02-28-11h41m09s756.png"

# T2.2
# VIDEO_PATH = "videos/T2.2/T2.2-rotated.mp4"
# LEARNING_IMAGE_PATH = "videos/T2.2/vlcsnap-2020-02-28-11h42m40s178.png"

# T2.3
VIDEO_PATH = "videos/demo.mp4"
LEARNING_IMAGE_PATH = "videos/capture.png"

# T3.1
# VIDEO_PATH = "videos/T3.1/T3.1-rotated.mp4"
# LEARNING_IMAGE_PATH = "videos/T3.1/vlcsnap-2020-02-28-11h43m42s674.png"

# T3.2
# VIDEO_PATH = "videos/T3.2/T3.2-rotated.mp4"
# LEARNING_IMAGE_PATH = "videos/T3.2/vlcsnap-2020-02-28-11h42m56s577.png"

# T3.3
# VIDEO_PATH = "videos/T3.3/T3.3-rotated.mp4"
# LEARNING_IMAGE_PATH = "videos/T3.3/vlcsnap-2020-02-28-11h42m56s577.png"

# VIDEO_PATH = "videoForBenchmark/Approche/video.mp4"
# LEARNING_IMAGE_PATH = "videoForBenchmark/Approche/reference.png"

class Client:
    """ Client simulator class."""

    def __init__(self):
        print("Connecting...")
        # self.sender = imagezmq.ImageSender(connect_to='tcp://10.1.162.31:5555')
        self.sender = imagezmq.ImageSender()

        if MODE_CAMERA:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            self.cap = cv2.VideoCapture(VIDEO_PATH)

        self.mode = MTEMode.NEUTRAL
        self.pov_id = 8
        self.learning_data = LearningData()
        time.sleep(2.0)  # allow camera sensor to warm up

    def run(self):
        """ Constant loop on camera or video, depending of the parameters."""
        if CAPTURE_DEMO:
            out = None

            if not os.path.exists(DEMO_FOLDER):
                os.makedirs(DEMO_FOLDER)

        # First size for MTE
        size = 400
        while self.cap.isOpened():
            # Sending
            success, full_image = self.cap.read()

            if not success:
                break

            image = imutils.resize(full_image, width=size)

            if CAPTURE_DEMO and out is None:
                demo_path = os.path.join(DEMO_FOLDER, 'demo_framing.avi')
                out = cv2.VideoWriter(demo_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
                                      10, (640*2, image.shape[0]))

            # print("Sending frame")
            if self.mode == MTEMode.VALIDATION_REFERENCE:
                if MODE_CAMERA:
                    image = full_image
                else:
                    image = cv2.imread(LEARNING_IMAGE_PATH)
                data = json.dumps({
                    "mode": self.mode.value,
                    "shape": image.shape
                })
            elif self.mode == MTEMode.INITIALIZE_MTE:
                temp = self.learning_data.to_dict()
                temp["mode"] = self.mode.value
                data = json.dumps(temp)
            elif self.mode == MTEMode.MOTION_TRACKING:
                data = json.dumps({
                    "mode": self.mode.value,
                    "shape": image.shape,
                    "id_ref" : self.learning_data.id_ref
                })
            elif self.mode == MTEMode.CLEAR_MTE:
                data = json.dumps({
                    "mode": self.mode.value,
                    "id_ref" : self.learning_data.id_ref
                })

            to_draw = full_image.copy()
            if self.mode != MTEMode.NEUTRAL:
                fps = FPS().start()
                reply = json.loads(self.sender.send_image(data, image).decode())
                fps.update()
                fps.stop()

                # Response
                if self.mode == MTEMode.VALIDATION_REFERENCE:
                    if ErrorLearning(reply["status"]) != ErrorLearning.SUCCESS:
                        print("Error during learning : {}".format(ErrorLearning(reply["status"])\
                            .name))
                    else:
                        print("Learning successfull")
                        self.learning_data.id_ref = -1
                        self.learning_data.mte_parameters = reply["mte_parameters"]
                elif self.mode == MTEMode.INITIALIZE_MTE:
                    if not reply["status"]:
                        print("Initialize successfull.")
                    else:
                        print("Initialize failed.")
                elif self.mode == MTEMode.MOTION_TRACKING:
                    response = reply
                    prev_size = size
                    if ErrorRecognition(reply["status"]) == \
                        ErrorRecognition.ENGINE_IS_NOT_INITIALIZED:
                        print("Error. You must first initialize the engine.")
                        self.mode = MTEMode.NEUTRAL
                    elif ErrorRecognition(reply["status"]) == ErrorRecognition.MISMATCH_REF:
                        print("The reference id is invalid.")
                        self.mode = MTEMode.NEUTRAL
                    else:
                        size = response["requested_image_size"][0]
                        to_draw = full_image
                        if response["flag"] == "ORANGE":
                            color_box = (0, 165, 255)
                        elif response["flag"] == "GREEN":
                            color_box = (0, 255, 0)
                        elif response["flag"] == "RED":
                            color_box = (0, 0, 255)
                        elif response["flag"] == "TARGET_LOST":
                            color_box = (50, 50, 50)
                        else:
                            color_box = (255, 255, 255)

                        cv2.putText(to_draw, "Size: {}".format(prev_size), (20, 20), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.putText(to_draw, response["flag"], \
                            (620, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

                        if response["flag"] != "RED" and \
                            response["flag"] != "TARGET_LOST":
                            cv2.putText(to_draw, "Trans. x: {:.2f}".format(response\
                                ["target_data"]["translations"][0]),\
                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            cv2.putText(to_draw, "Trans. y: {:.2f}".format(response\
                                ["target_data"]["translations"][1]), \
                                (220, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            cv2.putText(to_draw, "Scale x: {:.2f}".format(response\
                                ["target_data"]["scales"][0]),\
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            cv2.putText(to_draw, "Scale y: {:.2f}".format(response\
                                ["target_data"]["scales"][1]),\
                                (220, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        if not prev_size == size:
                            print("Change of size {} -> {}".format(prev_size, size))

                        if response["flag"] == "TARGET_LOST":
                            print("Flag TARGET_LOST")
                        elif response["flag"] == "RED":
                            print("Flag RED : {}".format(ErrorRecognition(response["status"]).name))
                        else:
                            print("Flag {} : {}".format(response["flag"], \
                                ErrorRecognition(response["status"]).name))
                            # Display target on image
                            cv2.putText(to_draw, "Direction: {}".format(response\
                                ["user_information"]), (20, 100),\
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            # Center point
                            x_coordinate = (full_image.shape[1]/image.shape[1]) * \
                                            (response["target_data"]["translations"][0]*\
                                            response["target_data"]["scales"][0] + image.shape[1]/3)
                            y_coordinate = (full_image.shape[0]/image.shape[0]) * \
                                            (response["target_data"]["translations"][1]*\
                                            response["target_data"]["scales"][1] + image.shape[0]/3)
                            center = cv2.KeyPoint(x_coordinate, y_coordinate, 8)
                            to_draw = cv2.drawKeypoints(to_draw, [center],\
                                                        np.array([]), (255, 0, 0), \
                                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                            upper_left_conner = (int(x_coordinate-full_image.shape[1]/3), \
                                                int(y_coordinate-full_image.shape[0]/3))
                            lower_right_corner = (int(x_coordinate+full_image.shape[1]/3), \
                                                int(y_coordinate+full_image.shape[0]/3))
                            to_draw = cv2.rectangle(to_draw, upper_left_conner,\
                                                    lower_right_corner, (255, 0, 0), thickness=3)

                            mean_scale = (response["target_data"]["scales"][0] + \
                                        response["target_data"]["scales"][1]) / 2
                            x_scaled = (full_image.shape[1]/image.shape[1]) * \
                                        (response["target_data"]["translations"][0]*\
                                            response["target_data"]["scales"][0] + image.shape[1]/3)
                            y_scaled = (full_image.shape[0]/image.shape[0]) * \
                                        (response["target_data"]["translations"][1]*\
                                            response["target_data"]["scales"][1] + image.shape[0]/3)
                            center_scaled = cv2.KeyPoint(x_scaled, y_scaled, 8)
                            to_draw = cv2.drawKeypoints(to_draw, [center_scaled], \
                                                        np.array([]), color_box, \
                                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                            upper_left_conner = (int(x_scaled-(full_image.shape[1]/3)*mean_scale), \
                                                int(y_scaled-(full_image.shape[0]/3)*mean_scale))
                            lower_right_corner = (int(x_scaled+(full_image.shape[1]/3)*mean_scale),\
                                                int(y_scaled+(full_image.shape[0]/3)*mean_scale))
                            to_draw = cv2.rectangle(to_draw, upper_left_conner,\
                                                    lower_right_corner, color_box, thickness=3)
                elif self.mode == MTEMode.CLEAR_MTE:
                    if reply["status"] == 0:
                        print("Clear successfull.")
                    else:
                        print("clear failed.")

                cv2.putText(to_draw, "FPS : {:.2f}".format(fps.fps()), (to_draw.shape[1]-120, 20), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow("Targetting", to_draw)
            key = cv2.waitKey(1)

            if self.mode == MTEMode.VALIDATION_REFERENCE or self.mode == MTEMode.INITIALIZE_MTE\
                or self.mode == MTEMode.CLEAR_MTE:
                self.mode = MTEMode.NEUTRAL
            if key == ord("1"):
                self.mode = MTEMode.NEUTRAL
            elif key == ord("2"):
                self.mode = MTEMode.VALIDATION_REFERENCE
                size = 400
            elif key == ord("3"):
                self.mode = MTEMode.INITIALIZE_MTE
            elif key == ord("4"):
                self.mode = MTEMode.MOTION_TRACKING
            elif key == ord("5"):
                self.mode = MTEMode.CLEAR_MTE
            elif key == ord("q"):
                sys.exit("User ended program.")
                if CAPTURE_DEMO:
                    out.release()
                    out = None

            # print(reply)

        # Release all
        if CAPTURE_DEMO and out is not None:
            out.release()

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    acd = Client()
    acd.run()
