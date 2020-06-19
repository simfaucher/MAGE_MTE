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
# VIDEO_PATH = "videos/T2.3/T2.3-rotated.mp4"
# LEARNING_IMAGE_PATH = "videos/T2.3/vlcsnap-2020-02-28-11h42m56s577.png"

# T3.1
# VIDEO_PATH = "videos/T3.1/T3.1-rotated.mp4"
# LEARNING_IMAGE_PATH = "videos/T3.1/vlcsnap-2020-02-28-11h43m42s674.png"

# T3.2
# VIDEO_PATH = "videos/T3.2/T3.2-rotated.mp4"
# LEARNING_IMAGE_PATH = "videos/T3.2/vlcsnap-2020-02-28-11h42m56s577.png"

# T3.3
# VIDEO_PATH = "videos/T3.3/T3.3-rotated.mp4"
# LEARNING_IMAGE_PATH = "videos/T3.3/vlcsnap-2020-02-28-11h42m56s577.png"

VIDEO_PATH = "videoForBenchmark/Approche/video.mp4"
LEARNING_IMAGE_PATH = "videoForBenchmark/Approche/reference.png"

class ACD:
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
        size = 380
        while self.cap.isOpened():
            # Sending
            success, full_image = self.cap.read()

            if not success:
                break

            # image_640 = cv2.resize(full_image, (size, int((size*9)/16)), interpolation=cv2.INTER_AREA)
            image_640 = imutils.resize(full_image, width=size)
            # image_640 = full_image

            if CAPTURE_DEMO and out is None:
                demo_path = os.path.join(DEMO_FOLDER, 'demo_framing.avi')
                out = cv2.VideoWriter(demo_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (640*2, image_640.shape[0]))

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

            if self.mode == MTEMode.FRAMING:
                reply, reply_image = self.sender.send_image_reqrep_image(json.dumps(data), image)
                reply = json.loads(reply)

                if image_640.shape[1] == reply_image.shape[1]:
                    stacked_images = np.hstack((image_640, reply_image))
                    cv2.imshow("Reply image framing", stacked_images)
                else:
                    cv2.imshow("Reply image framing", reply_image)

                if CAPTURE_DEMO and reply["framing"]["success"]:
                    out.write(stacked_images)
            else:
                reply = json.loads(self.sender.send_image(json.dumps(data), image).decode())

            # Response
            if self.mode == MTEMode.PRELEARNING:
                print("Number of keypoints: {}".format(reply["prelearning"]["nb_kp"]))
            elif self.mode == MTEMode.LEARNING:
                self.pov_id = reply["learning"]["id"]
                if self.pov_id == -1:
                    print("Echec de l'apprentissage.")
            elif self.mode == MTEMode.RECOGNITION:
                reco_data = reply["recognition"]
                prev_size = size
                response = reply["recognition"]["results"]
                size = response["size"]
                if not prev_size == size:
                    print("Changement de taille {} -> {}".format(prev_size, size))
                if reco_data["success"]:
                    print("Recognition OK")
                elif "sift_success" in reco_data and reco_data["sift_success"]:
                    print("Scale: {}, skew x: {}, skew y:{}, trans x: {}, trans y: {}".format(reco_data["scale"], \
                        reco_data["skew"]["x"], reco_data["skew"]["y"], \
                        reco_data["translation"]["x"], reco_data["translation"]["y"]))
                elif "vc_like_engine_success" in reco_data and reco_data["vc_like_engine_success"]:
                    print("Recognition VC-like success")
                else:
                    print("Recognition failed")
                if response["response"] == "TARGET_LOST":
                    print("Cible perdu")

                
                # Display target on image
                to_draw = full_image
                cv2.putText(to_draw, "{} matches".format(reco_data["nb_match"]), (160, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                (255, 255, 255), 2)
                if reco_data["success"] and not response["response"] == "TARGET_LOST":
                    # print(response)
                    cv2.putText(to_draw, "Trans. x: {:.2f}".format(response["shift_x"]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                        (255, 255, 255), 2)
                    cv2.putText(to_draw, "Trans. y: {:.2f}".format(response["shift_y"]), (160, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                        (255, 255, 255), 2)
                    cv2.putText(to_draw, "Scale x: {:.2f}".format(response["scale_h"]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                        (255, 255, 255), 2)
                    cv2.putText(to_draw, "Scale y: {:.2f}".format(response["scale_w"]), (160, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                        (255, 255, 255), 2)
                    x_coordinate = (full_image.shape[1]/prev_size+response["scale_h"])*response["shift_x"]
                    y_coordinate = (full_image.shape[1]/prev_size+response["scale_w"])*response["shift_y"]
                    # upper_left_conner = cv2.KeyPoint(x_coordinate, y_coordinate, 8)
                    # to_draw = cv2.drawKeypoints(to_draw, [upper_left_conner], np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    point = (int(x_coordinate), int(y_coordinate))
                    remaining_size_after_crop = (full_image.shape[1]*(2/3), full_image.shape[0]*(2/3))
                    second_point = (int(x_coordinate+remaining_size_after_crop[0]), int(y_coordinate+remaining_size_after_crop[1]))
                    to_draw = cv2.rectangle(to_draw, point, second_point, (255, 0, 0))
                cv2.imshow("Targetting", to_draw)
                cv2.waitKey(1)
            # elif mode == MTEMode.FRAMING:
            else:
                pass

            # Debugging
            debug_img = full_image.copy()
            cv2.imshow("Debug", debug_img)
            key = cv2.waitKey(1)

            if self.mode == MTEMode.LEARNING:
                self.mode = MTEMode.PRELEARNING

            if key == ord("1"):
                self.mode = MTEMode.PRELEARNING
            elif key == ord("2"):
                self.mode = MTEMode.LEARNING
            elif key == ord("3"):
                self.mode = MTEMode.RECOGNITION
            elif key == ord("4"):
                self.mode = MTEMode.FRAMING
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
    acd = ACD()
    acd.run()
