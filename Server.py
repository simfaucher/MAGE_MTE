"""
    Server side to catch a camera stream from a client
"""

import json
import numpy as np
import cv2
import imagezmq

from Domain.MTEMode import MTEMode
from Domain.LearningData import LearningData
from Domain.SiftData import SiftData
from Domain.MLData import MLData
from Repository import Repository

CROP_SIZE_HOR = 1/3
CROP_SIZE_VER = 1/3

FLANN_INDEX_KDTREE = 0
INDEX_PARAMS = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

SEARCH_PARAMS = dict(checks=50)
FLANN_THRESH = 0.7
MIN_MATCH_COUNT = 30

CAM_MATRIX = np.array([[954.16160543, 0., 635.29854945], \
    [0., 951.09864051, 359.47108905],  \
        [0., 0., 1.]])

class MTE:
    def __init__(self):
        print("Launching server")
        self.image_hub = imagezmq.ImageHub()
        # self.image_hub = imagezmq.ImageHub(open_port='tcp://192.168.43.39:5555')

        self.repo = Repository()
        self.sift = cv2.xfeatures2d.SIFT_create()

        self.learning_db = []
        self.last_learning_data = None

    def listen_images(self):
        while True:  # show streamed images until Ctrl-C
            msg, image = self.image_hub.recv_image()

            data = json.loads(msg)

            ret_data = {}

            mode = MTEMode(data["mode"])
            if mode == MTEMode.PRELEARNING:
                print("MODE prelearning")
                nb_kp = self.prelearning(image)
                # save_ref = "save_ref" in data and data["save_ref"]
                # ret_data["prelearning_pts"] = self.get_rectangle(0, image, force_new_ref=save_ref)
                ret_data["prelearning"] = {
                    "nb_kp": nb_kp
                }
            elif mode == MTEMode.LEARNING:
                print("MODE learning")
                self.learning(image)
            elif mode == MTEMode.RECOGNITION:
                pov_id = data["pov_id"]
                # print("MODE recognition")
                self.recognition(pov_id, image)
            # elif mode == MTEMode.FRAMING:
            else:
                pov_id = data["pov_id"]
                print("MODE framing")
                success, warped_image = self.framing(pov_id, image)

                # cv2.imshow("Warped image", warped_image)
                # cv2.waitKey(1)
            
            if mode == MTEMode.FRAMING:
                self.image_hub.send_reply_image(warped_image, json.dumps(ret_data))
            else:
                self.image_hub.send_reply(json.dumps(ret_data).encode())

    def init_get_rectangle(self, pov_id, image=None, force_new_ref=False):
        if force_new_ref:
            ref = image.copy()
            cv2.imwrite("ref.jpg", ref)
        elif self.latest_pov_id != pov_id:
            ref = cv2.imread("ref.jpg")

        ref = cv2.resize(ref, None, fx=0.5, fy=0.5)
        h_ref, w_ref = ref.shape[:2]
        self.ref = ref[int(h_ref/6): int(h_ref*5/6), int(w_ref/6): int(w_ref*5/6)]

        self.kp_ref, self.des_ref = self.sift.detectAndCompute(self.ref, None)

    def get_rectangle(self, pov_id, image, force_new_ref=False):
        self.init_get_rectangle(pov_id, image=image, force_new_ref=force_new_ref)

        image = cv2.resize(image, None, fx=0.5, fy=0.5)
        h_img, w_img = image.shape[:2]
        img = image[int(h_img/6): int(h_img*5/6), int(w_img/6): int(w_img*5/6)]
        kp_img, des_img = self.sift.detectAndCompute(img, None)

        FLANN_INDEX_KDTREE = 0
        INDEX_PARAMS = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

        SEARCH_PARAMS = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(INDEX_PARAMS, SEARCH_PARAMS)

        matches = flann.knnMatch(des_img, self.des_ref, k=2)

        FLANN_THRESH = 0.7
        # Need to draw only good matches, so create a mask
        # matchesMask = [[0, 0] for i in range(len(matches))]
        goodMatches = []

        # ratio test as per Lowe's paper
        for i, pair in enumerate(matches):
            try:
                m, n = pair
                if m.distance < FLANN_THRESH*n.distance:
                    goodMatches.append(m)
            except ValueError:
                pass

        # Homography
        MIN_MATCH_COUNT = 30
        # print("Matches found: %d/%d" % (len(goodMatches), MIN_MATCH_COUNT))

        if len(goodMatches) > MIN_MATCH_COUNT:
            dst_pts = np.float32([kp_img[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
            src_pts = np.float32([self.kp_ref[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w = self.ref.shape[:2]
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            img2 = cv2.polylines(img, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

        else:
            img2 = img.copy()
            matchesMask = None

        # Draw
        
        # DRAW_PARAMS = dict(matchColor=(0, 255, 0),
        #                 singlePointColor=(255, 0, 0),
        #                 matchesMask=matchesMask,
        #                 flags=0)

        # matching_result = cv2.drawMatches(img2, kp_img, self.ref.copy(), self.kp_ref, goodMatches, None, **DRAW_PARAMS)

        # cv2.imshow("Mathing", matching_result)
        # key = cv2.waitKey(1)

        # cv2.imshow("Reference", self.ref)
        # cv2.waitKey(1)

        if len(goodMatches) > MIN_MATCH_COUNT:
            ret_dst = []
            for pt in dst:
                ret_dst.append([(1/6 * w_img + pt[0][0])*2, (1/6 * h_img + pt[0][1])*2])
            return ret_dst
        else:
            return False

    def prelearning(self, image):
        # Renvoyer le nombre d'amers sur l'image envoyée
        img = self.crop_image(image)
        kp, _ = self.sift.detectAndCompute(img, None)
        return len(kp)

    def learning(self, full_image):
        # Enregistrement de l'image de référence en 640 pour SIFT + VC léger et 4K pour VCE
        learning_id = self.repo.save_new_pov(full_image)

        learning_data = self.repo.get_pov_by_id(learning_id)
        self.learning_db.append(learning_data)

    def recognition(self, pov_id, image):
        # Récupération d'une image, SIFT puis si validé VC léger avec mires auto. Si tout ok, envoi image 4K à VCE.
        learning_data = self.get_learning_data(pov_id)

        sift_success, H = self.get_homography_matrix(image, learning_data)

        #TODO: Logique de validation en fonction de la déformation de H
        # if sift_success:
        #     _, R, T, N = cv2.decomposeHomographyMat(H, CAM_MATRIX)
        #     print(R)
        if sift_success:
            h, w = image.shape[:2]
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H)

            debug_img = cv2.polylines(image, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

            cv2.imshow("Deformation", debug_img)

            # Print the deformation and scale
            print("scale x: {}, scale y: {}, skew x: {}, skew y: {}, tx: {}, ty: {}".format(round(H[0][0], 2), round(H[1][1], 2), round(H[0][1], 2), round(H[1][0], 2), round(H[0][2], 2), round(H[1][2], 2)))
            scale_x = H[0][0]
            scale_y = H[1][1]
            skew_x = H[0][1]
            skew_y = H[1][0]
            t_x = H[0][2]
            t_y = H[1][2]
        cv2.waitKey(1)

        #TODO: ML validation

        return sift_success

    def framing(self, pov_id, image):
        # Recadrage avec SIFT et renvoi de l'image
        learning_data = self.get_learning_data(pov_id)

        sift_success, H = self.get_homography_matrix(image, learning_data, \
            crop_image=False, dst_to_src=True)

        if sift_success:
            h, w = image.shape[:2]
            warped_image = cv2.warpPerspective(image, H, (w, h))
            return sift_success, warped_image
        else:
            return sift_success, image

    def crop_image(self, image):
        h, w = image.shape[:2]
        croped = image[int(h*CROP_SIZE_VER/2): int(h*(1-CROP_SIZE_VER/2)), \
            int(w*CROP_SIZE_HOR/2): int(w*(1-CROP_SIZE_HOR/2))]

        return croped

    def get_learning_data(self, pov_id):
        learning_data = None

        if self.last_learning_data is not None \
            and self.last_learning_data.id == pov_id:
            learning_data = self.last_learning_data
        else:
            items = [x for x in self.learning_db if x.id == pov_id]
            if len(items) > 0:
                learning_data = items[0]
            else:
                success, learning_data = self.repo.get_pov_by_id(pov_id)
                if not success:
                    raise Exception("No POV with id {}".format(pov_id))

        self.last_learning_data = learning_data

        # Learn SIFT data
        if learning_data.sift_data is None:
            croped = self.crop_image(learning_data.image_640)
            kp, des = self.sift.detectAndCompute(croped, None)

            learning_data.sift_data = SiftData(kp, des, croped)

        #TODO: Learn ML data

        return learning_data

    def get_homography_matrix(self, image, learning_data, crop_image=True, dst_to_src=False):
        if crop_image:
            img = self.crop_image(image)
        else:
            img = image

        h_img, w_img = img.shape[:2]
        kp_img, des_img = self.sift.detectAndCompute(img, None)

        flann = cv2.FlannBasedMatcher(INDEX_PARAMS, SEARCH_PARAMS)

        matches = flann.knnMatch(des_img, learning_data.sift_data.des, k=2)

        # Need to draw only good matches, so create a mask
        goodMatches = []

        # ratio test as per Lowe's paper
        for i, pair in enumerate(matches):
            try:
                m, n = pair
                if m.distance < FLANN_THRESH*n.distance:
                    goodMatches.append(m)
            except ValueError:
                pass

        # Homography
        # print("Matches found: %d/%d" % (len(goodMatches), MIN_MATCH_COUNT))

        success = len(goodMatches) > MIN_MATCH_COUNT

        if success:
            dst_pts = np.float32([kp_img[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
            src_pts = np.float32([learning_data.sift_data.kp[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

            if dst_to_src:
                H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            else:
                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        else:
            H = None

        return success, H

if __name__ == "__main__":
    mte = MTE()
    mte.listen_images()
