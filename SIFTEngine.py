"""
    Engine using SIFT to detect keypoints.
    Some parameters for validation are written here.
"""
import os
import sys
import time
from copy import deepcopy
import json
import numpy as np
import cv2

from Domain.MTEAlgo import MTEAlgo
from Domain.SiftData import SiftData
from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform

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
    HOMOGRAPHY_MIN_TRANS = -500
    HOMOGRAPHY_MAX_TRANS = 500

    def __init__(self, maxRansac):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.ransacmax = maxRansac
        self.resized_width = 380
        self.resized_height = 213
        self.img380 = None
        self.img640 = None
        self.display = None

    def learn(self, learning_data, crop_image=True, crop_margin=1/6):
        """Learn the sift keypoints of the image given through learning_data.
        It does not return a results but but changes values inside learning_data.
        """

        if learning_data.sift_data is None:
            dim = (self.resized_width, self.resized_height)
            img = cv2.resize(learning_data.full_image, dim, interpolation=cv2.INTER_AREA)
            keypoints_380, des_380, image_ref, kp_base_ransac = self.compute_sift(img, crop_image, crop_margin)
            self.img380 = image_ref
            # cv2.imwrite('ref_resize{}*{}.png'.format(self.resized_width, self.resized_height), image_ref)

            dim = (640, 360)
            img = cv2.resize(learning_data.full_image, dim, interpolation=cv2.INTER_AREA)
            keypoints_640, des_640, self.img640, _ = self.compute_sift(img, crop_image, crop_margin)
            # cv2.imwrite('ref_resize{}*{}.png'.format(dim[0], dim[1]), temp)

            dim = (1730, int(1730*9/16))
            img = cv2.resize(learning_data.full_image, dim, interpolation=cv2.INTER_AREA)
            keypoints_1730, des_1730, _, _ = self.compute_sift(img, crop_image, crop_margin)

            learning_data.sift_data = SiftData(keypoints_380, des_380, image_ref,\
                kp_base_ransac, keypoints_640, des_640, keypoints_1730, des_1730)

    def recognition(self, image, learning_data, modeAlgo):
        scale_x = 1
        scale_y = 1
        skew_x = 0
        skew_y = 0
        t_x = 0
        t_y = 0

        sift_success, src_pts, dst_pts, kp_img, des_img, good_matches, image = self.apply_sift(image, \
            learning_data.sift_data, debug=True, mode=modeAlgo)
        homography_success = False

        if sift_success:
            homography_matrix, mask = self.get_homography_matrix(src_pts, dst_pts, return_mask=True)
            matches_mask = mask.ravel().tolist()

            if image.shape[1] == 380:
                height, width = self.img380.shape[:2]
            else:
                height, width = self.img640.shape[:2]

            pts = np.float32([[0, 0], [0, height-1], [width-1, height-1], [width-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, homography_matrix)

            debug_img = cv2.polylines(image.copy(), [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

            # cv2.imshow("Deformation", debug_img)

            scale_x = float(homography_matrix[0][0])
            scale_y = float(homography_matrix[1][1])
            skew_x = float(homography_matrix[0][1])
            skew_y = float(homography_matrix[1][0])
            t_x = float(homography_matrix[0][2])
            t_y = float(homography_matrix[1][2])

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
                homography_matrix = self.get_homography_matrix(src_pts, dst_pts, dst_to_src=True)
                warped_image = cv2.warpPerspective(image, homography_matrix, (width, height))
                # on recup les kp en float + reshape pour passer le perspectiveTransform
                pos = []
                for i in range(len(kp_img)):
                     pos += [[kp_img[i].pt[0],kp_img[i].pt[1]]]
                pos = np.asarray(pos)
                pts = np.float32(pos).reshape(-1,1,2)
                new_pos = cv2.perspectiveTransform(pts, homography_matrix)
                if image.shape[1] == 380:
                    self.display = np.hstack((self.img380, warped_image))
                else:
                    self.display = np.hstack((self.img640, warped_image))

                cv2.imshow("Comparision", self.display)
                cv2.waitKey(1)
                # new_kp = []
                # for i in range(new_pos.shape[0]):
                #      new_kp += [cv2.KeyPoint(new_pos[i][0][0], new_pos[i][0][1], 1)]
                # warped_image = cv2.drawKeypoints(warped_image, new_kp, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            else:
                print("SIFT deformé")
                warped_image = image.copy()
        else:
            print("Pas assez de match pour lancer homographie: {}".format(len(good_matches)))
            warped_image = image.copy()
            for i in range (len(kp_img)):
                kp_img[i].size = 1
            warped_image = cv2.drawKeypoints(warped_image, kp_img, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return sift_success and homography_success, \
            (scale_x, scale_y), (skew_x, skew_y), (t_x, t_y), \
            warped_image, len(good_matches), len(kp_img)

    def get_homography_matrix(self, src_pts, dst_pts, dst_to_src=False, return_mask=False):
        if dst_to_src:
            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        else:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if return_mask:
            return H, mask
        else:
            return H

    def crop_image(self, image, crop_margin):
        h, w = image.shape[:2]
        croped = image[int(h*crop_margin): int(h*(1-crop_margin)), \
            int(w*crop_margin): int(w*(1-crop_margin))]

        return croped

    # Update : add keypointForRansac
    def compute_sift(self, image, crop_image, crop_margin=1/6):
        img = image
        if crop_image:
            img = self.crop_image(image, crop_margin)

        kp, des = self.sift.detectAndCompute(img, None)
        keypointForRansac = kp
        # print(kp[0].pt)

        return kp, des, img, keypointForRansac


    def apply_sift(self, image, sift_data, crop_image=False, crop_margin=1/6, debug=False, mode=MTEAlgo.SIFT_KNN):
        h, w = image.shape[:2]

        if w == 380:
            sift_data.switch_380()
        elif w == 640:
            sift_data.switch_640()
        else:
            sift_data.switch_1730()
        kp_img, des_img, image, kp_base = self.compute_sift(image, crop_image, crop_margin)

        if mode == MTEAlgo.SIFT_KNN:
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
        elif mode == MTEAlgo.SIFT_RANSAC:
            matches = match_descriptors(des_img, sift_data.des, cross_check=True)
            left = [kp_base[loop].pt[:] for loop in matches[:,0]]
            keypoints_left = np.asarray(left)
            right = [sift_data.kp_base[loop].pt[:] for loop in matches[:,1]]
            keypoints_right = np.asarray(right)
            np.random.seed(0)
            model, inliers = ransac(
                (keypoints_left, keypoints_right),
                ProjectiveTransform, min_samples=4,
                residual_threshold=4, max_trials=self.ransacmax
            )
            n_inliers = np.sum(inliers)
            # print(inliers)
            inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_left[inliers]]
            inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_right[inliers]]
            good_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]

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
            if mode == MTEAlgo.SIFT_KNN:
                dst_pts = np.float32([kp_img[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                src_pts = np.float32([sift_data.kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            elif mode == MTEAlgo.SIFT_RANSAC:
                dst_pts = np.float32([loop.pt for loop in inlier_keypoints_left]).reshape(-1, 1, 2)
                src_pts = np.float32([loop.pt for loop in inlier_keypoints_right]).reshape(-1, 1, 2)
        #     matches_mask = None
        #     debug_img = image.copy()

        # DRAW_PARAMS = dict(matchColor=(0, 255, 0), \
        #                 singlePointColor=(255, 0, 0), \
        #                 matchesMask=matches_mask, \
        #                 flags=0)
        # if mode == MTEAlgo.SIFT_KNN:
        #     matching_result = cv2.drawMatches(debug_img, kp_img, learning_data.sift_data.ref, learning_data.sift_data.kp, good_matches, None, **DRAW_PARAMS)
        # else:
        #     matching_result = cv2.drawMatches(debug_img, inlier_keypoints_left, sift_data.ref, inlier_keypoints_right, good_matches, None, **DRAW_PARAMS)
        # cv2.imshow("Matching result", matching_result)

        if debug:
            return success, src_pts, dst_pts, kp_img, des_img, good_matches, image
        else:
            return success, src_pts, dst_pts, image
