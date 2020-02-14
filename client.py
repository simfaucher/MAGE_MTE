"""
    Client side to stream a camera to a server
"""

import numpy as np
import socket
import json
import time
import cv2
import imutils
import imagezmq

from MTEMode import MTEMode

class ACD:
    def __init__(self):
        print("Connecting...")
        self.sender = imagezmq.ImageSender(connect_to='tcp://10.1.162.226:5555')
        # sender = imagezmq.ImageSender()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.mode = MTEMode.PRELEARNING
        self.id_pov = 0

        time.sleep(2.0)  # allow camera sensor to warm up

    def run(self):
        while True:  # send images as stream until Ctrl-C
            # Sending
            success, full_image = self.cap.read()

            if not success:
                continue

            image_640 = imutils.resize(full_image, width=640)

            data = {
                "mode": self.mode.value,
                "id_pov": self.id_pov
            }

            # print("Sending frame")
            reply = json.loads(self.sender.send_image(json.dumps(data), image_640).decode())

            # Response
            if self.mode == MTEMode.PRELEARNING:
                print("Number of keypoints: {}".format(reply["prelearning"]["nb_kp"]))
            elif self.mode == MTEMode.LEARNING:
                pass
            elif self.mode == MTEMode.RECOGNITION:
                pass
            # elif mode == MTEMode.FRAMING:
            else:
                pass

            # Debugging

            # if "prelearning_pts" in reply and reply["prelearning_pts"]:
            #     debug_img = cv2.polylines(image, [np.int32(reply["prelearning_pts"])], True, (0, 0, 255), 3, cv2.LINE_AA)
            # else:
            debug_img = full_image.copy()

            cv2.imshow("Debug", debug_img)
            key = cv2.waitKey(1)

            if key == ord("1"):
                self.mode = MTEMode.PRELEARNING
            elif key == ord("2"):
                self.mode = MTEMode.LEARNING
            elif key == ord("3"):
                self.mode = MTEMode.RECOGNITION
            elif key == ord("4"):
                self.mode = MTEMode.FRAMING

            # print(reply)

if __name__ == "__main__":
    acd = ACD()
    acd.run()
