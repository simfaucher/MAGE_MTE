#!/usr/bin/env python3

import socket
import threading
import socketserver
import cv2
import numpy as np

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):

    def setup(self):
        print("setup")

    def handle(self):
        # data = str(self.request.recv(1024), 'ascii')
        # cur_thread = threading.current_thread()
        # response = bytes("{}: {}".format(cur_thread.name, data), 'ascii')

        data = self.request.recv(1024)
        buff = np.fromstring(data, np.uint8)
        buff = buff.reshape((1, -1))

        image = cv2.imdecode(buff, cv2.IMREAD_COLOR)
        cv2.imshow("Render", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        self.request.sendall(b"Bien recu")

    def finish(self):
        print("finish")

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

if __name__ == "__main__":
    # Port 0 means to select an arbitrary unused port
    HOST, PORT = "localhost", 12330

    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    with server:
        ip, port = server.server_address

        # Start a thread with the server -- that thread will then start one
        # more thread for each request
        server_thread = threading.Thread(target=server.serve_forever)
        # Exit the server thread when the main thread terminates
        server_thread.daemon = True
        server_thread.start()
        print("Server loop running in thread:", server_thread.name)

        while True:
            pass

        server.shutdown()