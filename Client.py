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
import imagezmq
import Domain.ErrorLearning as ErrorLearning

from Domain.MTEMode import MTEMode
from imutils.video import FPS

CAPTURE_DEMO = False
DEMO_FOLDER = "demo/"

MODE_CAMERA = True
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
VIDEO_PATH = "videos/T2.3/T2.3-rotated.mp4"
LEARNING_IMAGE_PATH = "videos/T2.3/vlcsnap-2020-02-28-11h42m56s577.png"

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

        self.mode = MTEMode.PRELEARNING
        self.pov_id = 8

        time.sleep(2.0)  # allow camera sensor to warm up

    def run(self):
        if CAPTURE_DEMO:
            out = None

            if not os.path.exists(DEMO_FOLDER):
                os.makedirs(DEMO_FOLDER)
        size = 400
        equalize = "hsv"
        while self.cap.isOpened():
            # Sending
            success, full_image = self.cap.read()

            if not success:
                break
                        
            if equalize == "yuv":
                equalized = cv2.cvtColor(full_image, cv2.COLOR_BGR2YUV)
                equalized[:, :, 0] = cv2.equalizeHist(equalized[:, :, 0])
                cv2.imshow("yuv", equalized)
                full_image = cv2.cvtColor(equalized, cv2.COLOR_YUV2BGR)
            elif equalize == "hsv":
                equalized = cv2.cvtColor(full_image, cv2.COLOR_BGR2HSV)
                equalized[:, :, 2] = cv2.equalizeHist(equalized[:, :, 2])
                cv2.imshow("hsv", equalized)
                full_image = cv2.cvtColor(equalized, cv2.COLOR_HSV2BGR)

            image_640 = imutils.resize(full_image, width=size)
            # image_640 = full_image

            if CAPTURE_DEMO and out is None:
                demo_path = os.path.join(DEMO_FOLDER, 'demo_framing.avi')
                out = cv2.VideoWriter(demo_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
                                      10, (640*2, image_640.shape[0]))

            data = {
                "mode": self.mode.value,
                "pov_id": self.pov_id
            }

            # print("Sending frame")
            if self.mode == MTEMode.LEARNING:
                if MODE_CAMERA:
                    image = full_image
                else:
                    image = cv2.imread(LEARNING_IMAGE_PATH)
            else:
                image = image_640

            fps = FPS().start()
            reply = json.loads(self.sender.send_image(json.dumps(data), image).decode())
            fps.update()
            fps.stop()

            to_draw = full_image.copy()
            # Response
            if self.mode == MTEMode.PRELEARNING:
                print("Number of keypoints: {}".format(reply["prelearning"]["nb_kp"]))
            elif self.mode == MTEMode.LEARNING:
                self.pov_id = reply["learning"]["id"]
                if self.pov_id == -1 or not reply["learning"]["code"] == ErrorLearning.SUCCESS:
                    print("Failed to learn.")
            elif self.mode == MTEMode.RECOGNITION:
                reco_data = reply["recognition"]
                prev_size = size
                response = reply["recognition"]["results"]
                to_draw = full_image
                if response["response"] == "ORANGE":
                    color_box = (0, 165, 255)
                elif response["response"] == "GREEN":
                    color_box = (0, 255, 0)
                elif response["response"] == "RED":
                    color_box = (0, 0, 255)
                elif response["response"] == "TARGET_LOST":
                    color_box = (50, 50, 50)
                else:
                    color_box = (255, 255, 255)
                cv2.putText(to_draw, "Size: {}".format(prev_size), (20, 20), \
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(to_draw, "Nb kp: {:.2f}".format(reco_data["nb_kp"]), \
                    (220, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(to_draw, "{} matches".format(reco_data["nb_match"]), \
                    (420, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(to_draw, response["response"], \
                    (620, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)
                if reco_data["success"]:
                    cv2.putText(to_draw, "Dist 1: {:.2f}".format(reco_data["dist"][0]), (20, 40), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(to_draw, "Dist 2: {:.2f}".format(reco_data["dist"][1]), (220, 40), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(to_draw, "Dist 3: {:.2f}".format(reco_data["dist"][2]), (420, 40), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                if reco_data["nb_kp"] > 30 and not response["response"] == "RED" and \
                    not response["response"] == "TARGET_LOST":
                    cv2.putText(to_draw, "Trans. x: {:.2f}".format(response["shift_x"]), (20, 60), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(to_draw, "Trans. y: {:.2f}".format(response["shift_y"]), \
                        (220, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(to_draw, "Scale x: {:.2f}".format(response["scale_h"]), (20, 80), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(to_draw, "Scale y: {:.2f}".format(response["scale_w"]), (220, 80), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                size = response["size"]
                if not prev_size == size:
                    print("Change of size {} -> {}".format(prev_size, size))
                if reco_data["success"]:
                    print("Recognition OK")
                elif "sift_success" in reco_data and reco_data["sift_success"]:
                    print("Scale: {}, skew x: {}, skew y:{}, trans x: {}, trans y: {}".format(reco_data["scale"], \
                        reco_data["skew"]["x"], reco_data["skew"]["y"], \
                        reco_data["translation"]["x"], reco_data["translation"]["y"]))
                else:
                    print("Recognition failed")
                    cv2.putText(to_draw, "Homography failed.", (20, 100), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                if response["response"] == "TARGET_LOST":
                    print("Target lost")
                elif response["response"] == "RED":
                    print("RED : no homography")
                else:
                    # Display target on image
                    if reco_data["success"]:
                        # print(response)
                        cv2.putText(to_draw, "Direction: {}".format(response["direction"]), \
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.putText(to_draw, "Target", (220, 100), cv2.FONT_HERSHEY_SIMPLEX,\
                            0.5, (255, 0, 0), 2)
                        # Center point
                        x_coordinate = (full_image.shape[1]/image.shape[1]) * (response["shift_x"]*response["scale_w"] + image.shape[1]/3)
                        y_coordinate = (full_image.shape[0]/image.shape[0]) * (response["shift_y"]*response["scale_h"] + image.shape[0]/3)
                        center = cv2.KeyPoint(x_coordinate, y_coordinate, 8)
                        to_draw = cv2.drawKeypoints(to_draw, [center], np.array([]), (255, 0, 0), \
                                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                        upper_left_conner = (int(x_coordinate-full_image.shape[1]/3), \
                                             int(y_coordinate-full_image.shape[0]/3))
                        lower_right_corner = (int(x_coordinate+full_image.shape[1]/3), \
                                             int(y_coordinate+full_image.shape[0]/3))
                        to_draw = cv2.rectangle(to_draw, upper_left_conner,\
                                                lower_right_corner, (255, 0, 0), thickness=3)

                        mean_scale = (response["scale_w"] + response["scale_h"]) / 2
                        x_scaled = (full_image.shape[1]/image.shape[1]) * (response["shift_x"]*response["scale_w"] + image.shape[1]/3)
                        y_scaled = (full_image.shape[0]/image.shape[0]) * (response["shift_y"]*response["scale_h"] + image.shape[0]/3)
                        center_scaled = cv2.KeyPoint(x_scaled, y_scaled, 8)
                        to_draw = cv2.drawKeypoints(to_draw, [center_scaled], \
                                                    np.array([]), color_box, \
                                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                        upper_left_conner = (int(x_scaled-(full_image.shape[1]/3)*mean_scale), \
                                             int(y_scaled-(full_image.shape[0]/3)*mean_scale))
                        lower_right_corner = (int(x_scaled+(full_image.shape[1]/3)*mean_scale), \
                                             int(y_scaled+(full_image.shape[0]/3)*mean_scale))
                        to_draw = cv2.rectangle(to_draw, upper_left_conner,\
                                                lower_right_corner, color_box, thickness=3)

            else:
                pass

            cv2.putText(to_draw, "FPS : {:.2f}".format(fps.fps()), (to_draw.shape[1]-120, 20), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow("Targetting", to_draw)
            key = cv2.waitKey(1)

            if self.mode == MTEMode.LEARNING:
                self.mode = MTEMode.PRELEARNING

            if key == ord("1") or key == ord("&"):
                self.mode = MTEMode.PRELEARNING
            elif key == ord("2") or key == ord("é"):
                self.mode = MTEMode.LEARNING
                size = 400
            elif key == ord("3") or key == ord("\""):
                self.mode = MTEMode.RECOGNITION
            elif key == ord("q") or key == ord("Q"):
                sys.exit("User ended program.")
                if CAPTURE_DEMO:
                    out.release()
                    out = None
            elif key == ord("h") or key == ord("H"):
                equalize = "hsv"
                cv2.destroyWindow("yuv")
            elif key == ord("y") or key == ord("Y"):
                equalize = "yuv"
                cv2.destroyWindow("hsv")
            elif key == ord("n") or key == ord("N"):
                equalize = ""
                cv2.destroyWindow("yuv")
                cv2.destroyWindow("hsv")
            # print(reply)

        # Release all
        if CAPTURE_DEMO and out is not None:
            out.release()

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    acd = Client()
    acd.run()
