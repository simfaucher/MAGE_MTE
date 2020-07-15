from Client import Client

VIDEO_PATH = "videos/demo.mp4"
LEARNING_IMAGE_PATH = "videos/capture.png"

for i in range(100):
    client = Client()
    client.test(VIDEO_PATH, LEARNING_IMAGE_PATH)
