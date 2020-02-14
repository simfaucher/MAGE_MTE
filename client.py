"""
    Client side to stream a camera to a server
"""

import numpy as np
import socket
import json
import time
import cv2
from imutils.video import VideoStream
import imagezmq

print("Connecting...")
sender = imagezmq.ImageSender(connect_to='tcp://10.1.162.226:5555')
# sender = imagezmq.ImageSender()
 
rpi_name = socket.gethostname() # send RPi hostname with each image
# vs = VideoStream(usePiCamera=True).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)  # allow camera sensor to warm up

save_ref = False

while True:  # send images as stream until Ctrl-C
    image = vs.read()
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    print("Sending frame")
    msg = {
        "mode": 1
    }
    if save_ref:
        msg["save_ref"] = True
        save_ref = False

    reply = json.loads(sender.send_image(json.dumps(msg), image).decode())

    if "prelearning_pts" in reply and reply["prelearning_pts"]:
        debug_img = cv2.polylines(image, [np.int32(reply["prelearning_pts"])], True, (0, 0, 255), 3, cv2.LINE_AA)
    else:
        debug_img = image.copy()
    
    cv2.imshow("Debug", debug_img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        save_ref = True

    print(reply)
