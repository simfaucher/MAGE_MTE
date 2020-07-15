import csv
from Client import Client

VIDEO_PATH = "videos/demo.mp4"
LEARNING_IMAGE_PATH = "videos/capture.png"

test_csv = open('script_test.csv', 'w')
metrics = ['Success']
writer = csv.DictWriter(test_csv, fieldnames=metrics)
writer.writeheader()
for i in range(100):
    client = Client()
    results = client.test(VIDEO_PATH, LEARNING_IMAGE_PATH)
    writer.writerow({'Success' : results})
