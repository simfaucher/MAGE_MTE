#!/usr/bin/env python3

import socket
import sys
import time
import cv2

HOST, PORT = "localhost", 12345
data = " ".join(sys.argv[1:])

# Create a socket (SOCK_STREAM means a TCP socket)
while True:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))
        # Connect to server and send data
        sock.sendall(bytes(data + "\n", "utf-8"))

        # Receive data from the server and shut down
        received = str(sock.recv(1024), "utf-8")

        print("Sent:     {}".format(data))
        print("Received: {}".format(received))

        time.sleep(1)