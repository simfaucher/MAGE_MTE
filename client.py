# run this program on each RPi to send a labelled image stream
import socket
import time
from imutils.video import VideoStream
import imagezmq
import cv2
 
print("Connecting...")
sender = imagezmq.ImageSender(connect_to='tcp://192.168.42.129:5555')
 
rpi_name = socket.gethostname() # send RPi hostname with each image
# vs = VideoStream(usePiCamera=True).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)  # allow camera sensor to warm up


while True:  # send images as stream until Ctrl-C
    image = vs.read()
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    print("Sending frame")
    sender.send_image(rpi_name, image)