import csv
import os
import sys
import time
import threading
import subprocess
from Client import Client

VIDEO_PATH = "videos/demo.mp4"
LEARNING_IMAGE_PATH = "videos/capture.png"

test_csv = open('log_test'+'.csv', 'w')
metrics = ['Success']
writer = csv.DictWriter(test_csv, fieldnames=metrics)
writer.writeheader()

def timeout(writer_, results_, client_):
    print("script test time out")
    writer_.writerow({'Success' : results_})
    del client_

for i in range(100):
    # p=subprocess.Popen("notepad",shell=True)
    # time.sleep(1)
    # p.kill()
    # t = threading.Timer(40, timeout, [writer, False, client])
    # t.start()
    # results = client.test(VIDEO_PATH, LEARNING_IMAGE_PATH)
    # t.cancel()
    # writer.writerow({'Success' : results})
    process = subprocess.Popen(['timeout', '120', 'python', 'Client.py'], stderr=subprocess.PIPE)
    try:
        outs, errs = process.communicate(timeout=120)
        writer.writerow({'Success' : True})
    except subprocess.TimeoutExpired:
        process.kill()
        outs, errs = process.communicate()
        writer.writerow({'Success' : False})

    print(errs)
