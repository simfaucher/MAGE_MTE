#!/usr/bin/env python3

import socket
import sys
import time
import cv2

image = cv2.imread("images/loup-640.jpg")
image = cv2.resize(image, (320, 213))
# cv2.imshow("Render", image)
# cv2.waitKey(0)

HOST, PORT = "localhost", 12330
data = " ".join(sys.argv[1:])

# Create a socket (SOCK_STREAM means a TCP socket)
# while True:
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((HOST, PORT))
    # Connect to server and send data
    _, data = cv2.imencode('.jpg', image)
    for i in range(0, int(data.shape[0] / 1024) + 1):
        sock.sendall(data[i*1024: (i+1)*1024])

    # Receive data from the server and shut down
    received = str(sock.recv(1024), "utf-8")

    print("Sent:     {}".format(data))
    print("Received: {}".format(received))

    # time.sleep(1)