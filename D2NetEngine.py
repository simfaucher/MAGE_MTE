import os
import sys
import time
from copy import deepcopy
import json
import numpy as np
import cv2

from Domain.SiftData import SiftData

################### D2Net import ##################
import imageio

import torch
from tqdm import tqdm

import scipy
import scipy.io
import scipy.misc
from PIL import Image

from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform

from d2lib.model_test import D2Net
from d2lib.utils import preprocess_image
from d2lib.pyramid import process_multiscale

from d2lib.matchers import mutual_nn_matcher

from Domain.MTEAlgo import MTEAlgo

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print("Utilisation de cuda = {}".format(use_cuda))
#####################################################################################

CROP_SIZE_HOR = 1/3
CROP_SIZE_VER = 1/3

FLANN_INDEX_KDTREE = 0
INDEX_PARAMS = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

SEARCH_PARAMS = dict(checks=50)
FLANN_THRESH = 0.7


MIN_MATCH_COUNT = 7

D2REDUCTION = 1/3


class D2NetEngine:
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

    def __init__(self,max_edge,max_sum_edges,maxRansac,width,height):
        #Init d2net
        model_file="d2models/d2_tf_no_phototourism.pth"
        self.max_edge=max_edge
        self.max_sum_edges=max_sum_edges
        self.ransacmax = maxRansac
        self.resized_width = width
        self.resized_height = height
        self.preprocessing="caffe"
        self.multiscale=False
        use_relu=True
        print('==> Chargement du modele D2Net.')
        self.d2model = D2Net(
            model_file=model_file,
            use_relu=use_relu,
            use_cuda=use_cuda
        )
        print('==> Chargement terminer.')
        print('==> Pret pour client.')
        self.cpt = 0

    def learn(self, learning_data, crop_image=True, crop_margin=1/6):
        if learning_data.sift_data is None:
            kp, des, image_ref,kp_base_ransac = self.compute_d2(learning_data.full_image, crop_image, crop_margin)

            learning_data.sift_data = SiftData(kp, des, image_ref,kp_base_ransac)

    def recognition(self, image, learning_data,modeAlgo):
        scale_x = 1
        scale_y = 1
        skew_x = 0
        skew_y = 0
        t_x = 0
        t_y = 0

        d2succes, src_pts, dst_pts, kp_img, des_img, good_matches, image = self.apply_d2(image, \
            learning_data.sift_data, debug=True,mode=modeAlgo)
        homography_success = False

        if d2succes:
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

            scale_ok = D2NetEngine.HOMOGRAPHY_MIN_SCALE <= scale_x <= D2NetEngine.HOMOGRAPHY_MAX_SCALE \
                and D2NetEngine.HOMOGRAPHY_MIN_SCALE <= scale_y <= D2NetEngine.HOMOGRAPHY_MAX_SCALE
            skew_ok = 0 <= abs(skew_x) <= D2NetEngine.HOMOGRAPHY_MAX_SKEW \
                and 0 <= abs(skew_y) <= D2NetEngine.HOMOGRAPHY_MAX_SKEW
            translation_ok = D2NetEngine.HOMOGRAPHY_MIN_TRANS <= t_x <= D2NetEngine.HOMOGRAPHY_MAX_TRANS \
                and D2NetEngine.HOMOGRAPHY_MIN_TRANS <= t_y <= D2NetEngine.HOMOGRAPHY_MAX_TRANS

            homography_success = scale_ok and skew_ok and translation_ok

            if homography_success:
                print("Homographie valide")

                # Framing
                H = self.get_homography_matrix(src_pts, dst_pts, dst_to_src=True)
                warped_image = cv2.warpPerspective(image, H, (w, h))

            else:
                print("Homographie deformé")
                warped_image = image.copy()
        else:
            warped_image = image.copy()

        return d2succes and homography_success, \
            max(scale_x, scale_y), max(skew_x, skew_y), (t_x, t_y), \
            warped_image, \
            len(good_matches), \
            len(kp_img), \
            t_x+t_y, \
            skew_x+skew_y


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

    def compute_d2(self, image, crop_image, crop_margin=1/6):
        if crop_image:
            img = self.crop_image(image, crop_margin)
            # scale_percent = 25 # percent of original size
            # width = int(img.shape[1] * scale_percent / 100)
            # height = int(img.shape[0] * scale_percent / 100)
            # dim = (width, height)
            dim = (self.resized_width, self.resized_height)
            img = cv2.resize(img,dim, interpolation = cv2.INTER_AREA)
        else:
            dim = (self.resized_width, self.resized_height)
            img = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)

        # Setting up input image to use it in the CNN
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.repeat(image, 3, -1)

        resized_image = img
        if max(resized_image.shape) > self.max_edge:
            scale_percent = self.max_edge / max(resized_image.shape)
            width = int(resized_image.shape[1] * scale_percent)
            height = int(resized_image.shape[0] * scale_percent)
            dim = (width, height)
            resized_image = cv2.resize(resized_image,dim)
        if sum(resized_image.shape[: 2]) > self.max_sum_edges:
            scale_percent = self.max_sum_edges / sum(resized_image.shape[: 2])
            width = int(resized_image.shape[1] * scale_percent)
            height = int(resized_image.shape[0] * scale_percent)
            dim = (width, height)
            resized_image = cv2.resize(resized_image,dim)

        fact_i = img.shape[0] / resized_image.shape[0]
        fact_j = img.shape[1] / resized_image.shape[1]

        input_image = preprocess_image(
            resized_image,
            preprocessing=self.preprocessing
        )
        with torch.no_grad():
            if self.multiscale:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=device
                    ),
                    self.d2model
                )
            else:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=device
                    ),
                    self.d2model,
                    scales=[1]
                )

        # Input image coordinates
        keypoints[:, 0] *= fact_i
        keypoints[:, 1] *= fact_j

        # We zip the data in order to sort them by scores
        z = zip(scores, keypoints, descriptors)
        # We sort them by using the dimension 0
        # The sort is ascending
        s = sorted(z, key = lambda x: x[0])
        # Unzip
        tempScore,tempKp,tempDs = zip(*s)
        # Conversion back from tuple to array
        tempDesc = np.asarray(tempDs)
        tempKeyp = np.asarray(tempKp)
        # Reduction of size by D2REDUCTION factor, we are hoping to gain
        # time during the matching phase of the recognition
        descriptors = tempDesc[np.size(tempDesc,0) - int(np.size(tempDesc,0)*D2REDUCTION):np.size(tempDesc,0),:]
        keypoints = tempKeyp[np.size(tempKeyp,0) - int(np.size(tempKeyp,0)*D2REDUCTION):np.size(tempKeyp,0),:]

        # i, j -> u, v
        keypoints = keypoints[:, [1, 0, 2]]

        kp=[]
        for i in range(keypoints.shape[0]):
             kp += [cv2.KeyPoint(keypoints[i][0], keypoints[i][1], 1)]
        # print(kp[0].pt)

        return kp, descriptors, img,keypoints

    def apply_d2(self, image, sift_data, crop_image=False, crop_margin=1/6, debug=False,mode=MTEAlgo.D2NET_KNN):
        h, w = image.shape[:2]

        kp_img, des_img, image,kp_base = self.compute_d2(image, crop_image, crop_margin)

        if mode == MTEAlgo.D2NET_KNN :
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
            print(len(good_matches))
        elif mode == MTEAlgo.D2NET_RANSAC :
            ########################### Brute force sur les descripeurs + ransac (random samples) #################################################
            matches = match_descriptors(des_img, sift_data.des, cross_check=True)
            keypoints_left = kp_base[matches[:, 0], : 2]
            keypoints_right = sift_data.kp_base[matches[:, 1], : 2]
            np.random.seed(0)
            model, inliers = ransac(
                (keypoints_left, keypoints_right),
                ProjectiveTransform, min_samples=4,
                residual_threshold=4, max_trials=self.ransacmax
            )
            n_inliers = np.sum(inliers)
            print(keypoints_left[inliers])
            inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_left[inliers]]
            inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_right[inliers]]
            good_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
            # print("{} et {}".format(good_matches[0].queryIdx,good_matches[0].trainIdx))
        else:
            ########################### Match knn GPU #######################
            descriptors1 = torch.from_numpy(des_img).to(device)
            descriptors2 = torch.from_numpy(sift_data.des).to(device)
            good_matches=mutual_nn_matcher(descriptors1, descriptors2)
            print(len(good_matches))

        # Homography
        # print("Matches found: %d/%d" % (len(goodMatches), MIN_MATCH_COUNT))

        success = len(good_matches) > MIN_MATCH_COUNT

        dst_pts = []
        src_pts = []
        if success:
            if mode == MTEAlgo.D2NET_KNN:
                dst_pts = np.float32([kp_img[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                src_pts = np.float32([sift_data.kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            elif mode == MTEAlgo.D2NET_RANSAC:
                dst_pts = np.float32([loop.pt for loop in inlier_keypoints_left]).reshape(-1, 1, 2)
                src_pts = np.float32([loop.pt for loop in inlier_keypoints_right]).reshape(-1, 1, 2)
            else :
                dst_pts = np.float32([kp_img[m[0]].pt for m in good_matches]).reshape(-1, 1, 2)
                src_pts = np.float32([sift_data.kp[m[1]].pt for m in good_matches]).reshape(-1, 1, 2)

        # matches_mask = None
        # debug_img = image.copy()
        #
        # DRAW_PARAMS = dict(matchColor=(0, 255, 0), \
        #                 singlePointColor=(255, 0, 0), \
        #                 matchesMask=matches_mask, \
        #                 flags=0)
        # if mode == MTEAlgo.D2NET_KNN:
        #     matching_result = cv2.drawMatches(debug_img, kp_img, sift_data.ref, sift_data.kp, good_matches, None, **DRAW_PARAMS)
        # else:
        #     matching_result = cv2.drawMatches(debug_img, inlier_keypoints_left, sift_data.ref, inlier_keypoints_right, good_matches, None, **DRAW_PARAMS)
        #
        # cv2.imwrite("framing/matching {}.png".format(self.cpt), matching_result)
        self.cpt += 1

        if debug:
            return success, src_pts, dst_pts, kp_img, des_img, good_matches, image
        else:
            return success, src_pts, dst_pts, image
