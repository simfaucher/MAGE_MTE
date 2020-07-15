#!/usr/bin/env python3

"""
    Finds lines in an image using multiple algorithms.

    Project Nose Landing Gear Video Measurement for ATR
    Created on Mon Oct 21 2019 by Frank Ben Zaquin, Fabien Monniot
    Copyright (c) 2019 Altran Technologies
"""

import numpy as np
import cv2
from ML.Domain.ImageFilterType import ImageFilterType

# Parameters
LSD_THICKNESS = 3

CANNY_MIN_THRESH = 50
CANNY_MAX_THRESH = 98

EMBOSS_K = 4
EMBOSS_KERNEL = np.array([[0,0,0,0,0],[0,0,0,0,0],[EMBOSS_K,EMBOSS_K,0,-EMBOSS_K,-EMBOSS_K],[0,0,0,0,0],[0,0,0,0,0]])
EMBOSS_THRESH = 120

KIRSCH_K1 = 5
KIRSCH_K2 = 3
KIRSCH_THRESH = 130

KIRSCH_KERNEL_1 = np.array([[KIRSCH_K1,KIRSCH_K1,KIRSCH_K1],[-KIRSCH_K2,0,-KIRSCH_K2],[-KIRSCH_K2,-KIRSCH_K2,-KIRSCH_K2]])
KIRSCH_KERNEL_2 = np.array([[KIRSCH_K1,KIRSCH_K1,-KIRSCH_K2],[KIRSCH_K1,0,-KIRSCH_K2],[-KIRSCH_K2,-KIRSCH_K2,-KIRSCH_K2]])
KIRSCH_KERNEL_3 = np.array([[KIRSCH_K1,-KIRSCH_K2,-KIRSCH_K2],[KIRSCH_K1,0,-KIRSCH_K2],[KIRSCH_K1,-KIRSCH_K2,-KIRSCH_K2]])
KIRSCH_KERNEL_4 = np.array([[-KIRSCH_K2,-KIRSCH_K2,-KIRSCH_K2],[KIRSCH_K1,0,-KIRSCH_K2],[KIRSCH_K1,KIRSCH_K1,-KIRSCH_K2]])
KIRSCH_KERNEL_5 = np.array([[-KIRSCH_K2,-KIRSCH_K2,-KIRSCH_K2],[-KIRSCH_K2,0,-KIRSCH_K2],[KIRSCH_K1,KIRSCH_K1,KIRSCH_K1]])
KIRSCH_KERNEL_6 = np.array([[-KIRSCH_K2,-KIRSCH_K2,-KIRSCH_K2],[-KIRSCH_K2,0,KIRSCH_K1],[-KIRSCH_K2,KIRSCH_K1,KIRSCH_K1]])
KIRSCH_KERNEL_7 = np.array([[-KIRSCH_K2,-KIRSCH_K2,KIRSCH_K1],[-KIRSCH_K2,0,KIRSCH_K1],[-KIRSCH_K2,-KIRSCH_K2,KIRSCH_K1]])
KIRSCH_KERNEL_8 = np.array([[-KIRSCH_K2,KIRSCH_K1,KIRSCH_K1],[-KIRSCH_K2,0,KIRSCH_K1],[-KIRSCH_K2,-KIRSCH_K2,-KIRSCH_K2]])

BINARISATION_WIDTH_PERCENT = 0      # 0 <= BINARISATION_WIDTH_PERCENT < 50 (% on each side)
BINARISATION_HEIGHT_PERCENT = 0    # 0 <= BINARISATION_HEIGHT_PERCENT < 50 (% on each side)

class LinesDetector():
    def __init__(self, input_image, algo):
        self.input_image = input_image
        self.algo = algo

        self.dilation_kernel = np.ones((3, 3), np.uint8)

    def detect(self):
        # Initialize mask
        height, width = self.input_image.shape[:2]
        grey = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2GRAY)
        # normalized = cv2.equalizeHist(grey)

        if self.algo == ImageFilterType.LSD:
            # Find lines with LSD
            lsd = cv2.createLineSegmentDetector(0)
            lines = lsd.detect(grey)[0]

            mask = self.get_lsd_mask(grey, lines, LSD_THICKNESS)
        elif self.algo == ImageFilterType.CANNY:
            # Find lines with Canny
            smoothed = cv2.GaussianBlur(grey, (7, 7), 2) #TODO: supprimer/garder ?
            mask = cv2.Canny(smoothed, CANNY_MIN_THRESH, CANNY_MAX_THRESH)
            mask = cv2.dilate(mask, self.dilation_kernel, iterations=1)
        elif self.algo == ImageFilterType.EMBOSS:
            # Find lines with embossing
            mask = cv2.filter2D(grey, -1, EMBOSS_KERNEL)
        elif self.algo == ImageFilterType.KIRSCH:
            # Find lines with Kirsch
            kirsch1 = (cv2.filter2D(grey, -1, KIRSCH_KERNEL_1))
            kirsch2 = (cv2.filter2D(grey, -1, KIRSCH_KERNEL_2))
            kirsch3 = (cv2.filter2D(grey, -1, KIRSCH_KERNEL_3))
            kirsch4 = (cv2.filter2D(grey, -1, KIRSCH_KERNEL_4))
            kirsch5 = (cv2.filter2D(grey, -1, KIRSCH_KERNEL_5))
            kirsch6 = (cv2.filter2D(grey, -1, KIRSCH_KERNEL_6))
            kirsch7 = (cv2.filter2D(grey, -1, KIRSCH_KERNEL_7))
            kirsch8 = (cv2.filter2D(grey, -1, KIRSCH_KERNEL_8))

            mask = np.maximum(kirsch1, \
                np.maximum(kirsch2, \
                    np.maximum(kirsch3, \
                        np.maximum(kirsch4, \
                            np.maximum(kirsch5, \
                                np.maximum(kirsch6, \
                                    np.maximum(kirsch7, kirsch8)))))))
        elif self.algo == ImageFilterType.BINARISATION:
            mask = self.binarisation_whole_mean(grey)
        elif self.algo == ImageFilterType.POSTERISATION:
            mask = self.input_image.copy()
            mask[mask >= 128] = 255
            mask[mask < 128] = 0
        else:
            mask = grey

        return mask

    def get_lsd_mask(self, image, lines, thickness):
        """ Get the mask from the LSD output lines """

        height, width = image.shape[:2]
        mask = np.zeros((height, width), np.uint8)

        for line in lines:
            start = (int(line[0][0] - 1), int(line[0][1] - 1))
            end = (int(line[0][2] - 1), int(line[0][3] - 1))

            cv2.line(mask, start, end, 255, thickness)

        return mask
    
    def binarisation(self, image):
        height, width = image.shape[:2]
        pt_tl = (int(BINARISATION_WIDTH_PERCENT * width / 100), int(BINARISATION_HEIGHT_PERCENT * height / 100))
        pt_br = (width - pt_tl[0], height - pt_tl[1])

        roi = image[pt_tl[1]: pt_br[1], pt_tl[0]: pt_br[0]]
        mean = np.mean(roi)

        mask = np.zeros((height, width), np.uint8)
        mask[image > mean] = 255
        
        return mask

    def binarisation_whole_mean(self, image):
        height, width = image.shape[:2]
        mean = np.mean(image)

        mask = np.zeros((height, width), np.uint8)
        mask[image > mean] = 255
        
        return mask
