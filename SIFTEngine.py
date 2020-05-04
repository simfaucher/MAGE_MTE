import os
import sys
import time
from copy import deepcopy
import json
import numpy as np
import cv2

from Domain.SiftData import SiftData

CROP_SIZE_HOR = 1/3
CROP_SIZE_VER = 1/3

FLANN_INDEX_KDTREE = 0
INDEX_PARAMS = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

SEARCH_PARAMS = dict(checks=50)
FLANN_THRESH = 0.7
MIN_MATCH_COUNT = 30

class SIFTEngine:
    # HOMOGRAPHY_MIN_SCALE = 0.75
    # HOMOGRAPHY_MAX_SCALE = 1.25
    # HOMOGRAPHY_MAX_SKEW = 0.13
    # HOMOGRAPHY_MIN_TRANS = 0
    # HOMOGRAPHY_MAX_TRANS = 50
    HOMOGRAPHY_MIN_SCALE = 0.0
    HOMOGRAPHY_MAX_SCALE = 3
    HOMOGRAPHY_MAX_SKEW = 1
    HOMOGRAPHY_MIN_TRANS = 0
    HOMOGRAPHY_MAX_TRANS = 500

    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()

    def learn(self, learning_data, crop_image=True):
        if learning_data.sift_data is None:
            kp, des, image_ref = self.compute_sift(learning_data.image_640, crop_image)

            learning_data.sift_data = SiftData(kp, des, image_ref)
    
    def recognition(self, image, learning_data):
        scale_x = 1
        scale_y = 1
        skew_x = 0
        skew_y = 0
        t_x = 0
        t_y = 0

        sift_success, src_pts, dst_pts, kp_img, des_img, good_matches, image = self.apply_sift(image, \
            learning_data.sift_data, debug=True)
        homography_success = False

        if sift_success:
            H, mask = self.get_homography_matrix(src_pts, dst_pts, return_mask=True)
            matches_mask = mask.ravel().tolist()

            h, w = learning_data.sift_data.ref.shape[:2]
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H)

            debug_img = cv2.polylines(image.copy(), [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

            # cv2.imshow("Deformation", debug_img)

            scale_x = float(H[0][0])
            scale_y = float(H[1][1])
            skew_x = float(H[0][1])
            skew_y = float(H[1][0])
            t_x = float(H[0][2])
            t_y = float(H[1][2])

            scale_ok = SIFTEngine.HOMOGRAPHY_MIN_SCALE <= scale_x <= SIFTEngine.HOMOGRAPHY_MAX_SCALE \
                and SIFTEngine.HOMOGRAPHY_MIN_SCALE <= scale_y <= SIFTEngine.HOMOGRAPHY_MAX_SCALE
            skew_ok = 0 <= abs(skew_x) <= SIFTEngine.HOMOGRAPHY_MAX_SKEW \
                and 0 <= abs(skew_y) <= SIFTEngine.HOMOGRAPHY_MAX_SKEW
            translation_ok = SIFTEngine.HOMOGRAPHY_MIN_TRANS <= t_x <= SIFTEngine.HOMOGRAPHY_MAX_TRANS \
                and SIFTEngine.HOMOGRAPHY_MIN_TRANS <= t_y <= SIFTEngine.HOMOGRAPHY_MAX_TRANS

            homography_success = scale_ok and skew_ok and translation_ok

            if homography_success:
                print("SIFT valide")

                # Framing
                H = self.get_homography_matrix(src_pts, dst_pts, dst_to_src=True)
                warped_image = cv2.warpPerspective(image, H, (w, h))

            else:
                print("SIFT deformé")
                warped_image = image.copy()
        else:
            warped_image = image.copy()

        return sift_success and homography_success, \
            max(scale_x, scale_y), max(skew_x, skew_y), (t_x, t_y), \
            warped_image
            

    def get_homography_matrix(self, src_pts, dst_pts, dst_to_src=False, return_mask=False):
        if dst_to_src:
            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        else:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if return_mask:
            return H, mask
        else:
            return H

    def crop_image(self, image):
        h, w = image.shape[:2]
        croped = image[int(h*CROP_SIZE_VER/2): int(h*(1-CROP_SIZE_VER/2)), \
            int(w*CROP_SIZE_HOR/2): int(w*(1-CROP_SIZE_HOR/2))]

        return croped

    def compute_sift(self, image, crop_image=True):
        if crop_image:
            img = self.crop_image(image)
        else:
            img = image

        kp, des = self.sift.detectAndCompute(img, None)

        return kp, des, img


    def apply_sift(self, image, sift_data, crop_image=True, debug=False):
        h, w = image.shape[:2]

        kp_img, des_img, image = self.compute_sift(image, crop_image)

        flann = cv2.FlannBasedMatcher(INDEX_PARAMS, SEARCH_PARAMS)

        matches = flann.knnMatch(des_img, sift_data.des, k=2)

        # Need to draw only good matches, so create a mask
        good_matches = []

        # ratio test as per Lowe's paper
        for i, pair in enumerate(matches):
            try:
                m, n = pair
                if m.distance < FLANN_THRESH*n.distance:
                    good_matches.append(m)
            except ValueError:
                pass

        # Add crop
        # if crop_image:
        #     for kp in kp_img:
        #         kp.pt = (kp.pt[0] + w * CROP_SIZE_HOR, kp.pt[1] + h * CROP_SIZE_VER)

        # Homography
        # print("Matches found: %d/%d" % (len(goodMatches), MIN_MATCH_COUNT))

        success = len(good_matches) > MIN_MATCH_COUNT

        dst_pts = []
        src_pts = []
        if success:
            dst_pts = np.float32([kp_img[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            src_pts = np.float32([sift_data.kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        #     matches_mask = None
        #     debug_img = image.copy()

        # DRAW_PARAMS = dict(matchColor=(0, 255, 0), \
        #                 singlePointColor=(255, 0, 0), \
        #                 matchesMask=matches_mask, \
        #                 flags=0)

        # matching_result = cv2.drawMatches(debug_img, kp_img, learning_data.sift_data.ref, learning_data.sift_data.kp, good_matches, None, **DRAW_PARAMS)
        # cv2.imshow("Matching result", matching_result)

        if debug:
            return success, src_pts, dst_pts, kp_img, des_img, good_matches, image
        else:
            return success, src_pts, dst_pts, image
