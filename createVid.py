import cv2
import numpy as np
import glob

#This function create a video from a single image

img_array = []
img = cv2.imread('./videoForBenchmark/flou/reference.jpg')
height, width, layers = img.shape
size = (width,height)
for i in range (0,300):
    print("check")
    img_array.append(img)


out = cv2.VideoWriter('./videoForBenchmark/flou/video.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
