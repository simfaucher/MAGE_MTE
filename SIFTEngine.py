"""
    Engine using SIFT to detect keypoints.
    Some parameters for validation are written here.
"""
import numpy as np
import cv2

from Domain.MTEAlgo import MTEAlgo
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
    HOMOGRAPHY_MIN_SCALE = 0.75
    HOMOGRAPHY_MAX_SCALE = 1.25
    HOMOGRAPHY_MAX_SKEW = 0.13
    HOMOGRAPHY_MIN_TRANS = -25
    HOMOGRAPHY_MAX_TRANS = 50
    # HOMOGRAPHY_MIN_SCALE = 0.0
    # HOMOGRAPHY_MAX_SCALE = 3
    # HOMOGRAPHY_MAX_SKEW = 1
    # HOMOGRAPHY_MIN_TRANS = -500
    # HOMOGRAPHY_MAX_TRANS = 500

    def __init__(self, maxRansac):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.ransacmax = maxRansac
        self.img_small = None
        self.img_medium = None
        self.display = None
        self.width_small = None
        self.width_medium = None
        self.width_large = None
        self.resized_width = None
        self.resized_height = None
        self.format_resolution = None
    
    def set_parameters(self, width_small, width_medium, width_large, format_resolution):
        self.width_small = width_small
        self.width_medium = width_medium
        self.width_large = width_large
        self.resized_width = width_small
        self.resized_height = int(self.resized_width * (1/format_resolution))
        self.format_resolution = format_resolution

    def learn(self, image_ref, learning_data, crop_image=True, crop_margin=1/6):
        """Learn the sift keypoints of the image given through learning_data.
        It does not return a result but but changes values inside learning_data.
        """

        if learning_data.id_ref is None:
            dim = (self.resized_width, self.resized_height)
            img = cv2.resize(image_ref, dim, interpolation=cv2.INTER_AREA)
            keypoints_small, des_small, self.img_small = self.compute_sift(img, crop_image, crop_margin)

            dim = (self.width_medium, int(self.width_medium * (1/self.format_resolution)))
            img = cv2.resize(image_ref, dim, interpolation=cv2.INTER_AREA)
            keypoints_medium, des_medium, self.img_medium = self.compute_sift(img, crop_image, crop_margin)

            dim = (self.width_large, int(self.width_large * (1/self.format_resolution)))
            img = cv2.resize(image_ref, dim, interpolation=cv2.INTER_AREA)
            keypoints_large, des_large, _ = self.compute_sift(img, crop_image, crop_margin)

            learning_data.fill_with_engine_for_learning(self.format_resolution, keypoints_small, des_small, \
                keypoints_medium, des_medium, keypoints_large, des_large)

    def recognition(self, image_init, learning_data, modeAlgo):
        scale_x = 1
        scale_y = 1
        skew_x = 0
        skew_y = 0
        t_x = 0
        t_y = 0

        sift_success, src_pts, dst_pts, kp_img, des_img, good_matches, image = self.apply_sift(image_init, \
            learning_data.mte_parameters, debug=True, mode=modeAlgo)
        homography_success = False

        if sift_success:
            homography_matrix, mask = self.get_homography_matrix(src_pts, dst_pts, return_mask=True)
            matches_mask = mask.ravel().tolist()

            height, width = self.crop_image(image_init, 1/6).shape[:2]

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
            # translation_ok = SIFTEngine.HOMOGRAPHY_MIN_TRANS <= t_x <= SIFTEngine.HOMOGRAPHY_MAX_TRANS \
            #     and SIFTEngine.HOMOGRAPHY_MIN_TRANS <= t_y <= SIFTEngine.HOMOGRAPHY_MAX_TRANS

            # homography_success = scale_ok and skew_ok and translation_ok
            homography_success = scale_ok and skew_ok

            if homography_success:
                print("SIFT valid")

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
                # if image.shape[1] == self.width_small:
                #     self.display = np.hstack((self.img_small, warped_image))
                # else:
                #     self.display = np.hstack((self.img_medium, warped_image))

                # cv2.imshow("Comparison", self.display)
                # cv2.waitKey(1)

            else:
                print("SIFT distored")
                warped_image = image.copy()
        else:
            print("Not enough match for homography: {}".format(len(good_matches)))
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

    def compute_sift(self, image, crop_image, crop_margin=1/6):
        img = image
        if crop_image:
            img = self.crop_image(image, crop_margin)

        keypoints, des = self.sift.detectAndCompute(img, None)

        return keypoints, des, img


    def apply_sift(self, image, sift_data, crop_image=False, crop_margin=1/6, debug=False, mode=MTEAlgo.SIFT_KNN):
        h, w = image.shape[:2]

        if w == self.width_small:
            keypoints = sift_data["size_small"]["keypoints"]
            descriptors = sift_data["size_small"]["descriptors"]
        elif w == self.width_medium:
            keypoints = sift_data["size_medium"]["keypoints"]
            descriptors = sift_data["size_medium"]["descriptors"]
        else:
            keypoints = sift_data["size_large"]["keypoints"]
            descriptors = sift_data["size_large"]["descriptors"]

        kp_img, des_img, image = self.compute_sift(image, crop_image, crop_margin)
        kp_base = cv2.KeyPoint_convert(kp_img)

        if mode == MTEAlgo.SIFT_KNN:
            flann = cv2.FlannBasedMatcher(INDEX_PARAMS, SEARCH_PARAMS)

            matches = flann.knnMatch(des_img, descriptors, k=2)

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
            matches = match_descriptors(des_img, descriptors, cross_check=True)
            left = [kp_base[loop].pt[:] for loop in matches[:, 0]]
            keypoints_left = np.asarray(left)
            right = [cv2.KeyPoint_convert(keypoints)[loop].pt[:] for loop in matches[:, 1]]
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

        success = len(good_matches) > MIN_MATCH_COUNT

        dst_pts = []
        src_pts = []
        if success:
            if mode == MTEAlgo.SIFT_KNN:
                dst_pts = np.float32([kp_img[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                src_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
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
