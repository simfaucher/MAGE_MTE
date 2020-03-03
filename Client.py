"""
    Client side to stream a camera to a server
"""

import sys
import time
import socket
import json
import numpy as np
import cv2
import imutils
import imagezmq

from Domain.MTEMode import MTEMode

CAPTURE_DEMO = False

MODE_CAMERA = False
MODE_VIDEO = not MODE_CAMERA

# T1.1
# VIDEO_PATH = "videos/T1.1/VID_20200302_144048.mp4"
# LEARNING_IMAGE_PATH = "videos/T1.1/vlcsnap-2020-03-02-15h59m47s327.png"

# T1.2
# VIDEO_PATH = "videos/T1.2/Zoom/VID_20200302_144327.mp4"
# LEARNING_IMAGE_PATH = "videos/T1.2/Zoom/vlcsnap-2020-03-02-16h00m31s968.png"

# T1.3
VIDEO_PATH = "videos/T1.3/Zoom/VID_20200302_144507.mp4"
LEARNING_IMAGE_PATH = "videos/T1.3/Zoom/vlcsnap-2020-03-02-16h01m23s741.png"

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

        while self.cap.isOpened():
            # Sending
            success, full_image = self.cap.read()

            if not success:
                continue

            image_640 = imutils.resize(full_image, width=640)

            if CAPTURE_DEMO and out is None:
                out = cv2.VideoWriter('demo_framing.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (640*2, image_640.shape[0]))

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
                stacked_images = np.hstack((image_640, reply_image))
                cv2.imshow("Reply image framing", stacked_images)
                if CAPTURE_DEMO and reply["framing"]["success"]:
                    out.write(stacked_images)
            else:
                reply = json.loads(self.sender.send_image(json.dumps(data), image).decode())

            # Response
            if self.mode == MTEMode.PRELEARNING:
                print("Number of keypoints: {}".format(reply["prelearning"]["nb_kp"]))
            elif self.mode == MTEMode.LEARNING:
                self.pov_id = reply["learning"]["id"]
            elif self.mode == MTEMode.RECOGNITION:
                reco_data = reply["recognition"]
                if reco_data["success"]:
                    print("Recognition OK")
                    # TODO: Envoi image 4K à VCE.
                elif reco_data["sift_success"]:
                    print("Scale: {}, skew x: {}, skew y:{}, trans x: {}, trans y: {}".format(reco_data["scale"], \
                        reco_data["skew"]["x"], reco_data["skew"]["y"], \
                        reco_data["translation"]["x"], reco_data["translation"]["y"]))
                else:
                    print("Recognition failed")
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

            # print(reply)

if __name__ == "__main__":
    acd = ACD()
    acd.run()
