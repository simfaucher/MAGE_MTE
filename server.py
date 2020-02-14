"""
    Server side to catch a camera stream from a client
"""

from enum import Enum
import json
import numpy as np
import cv2
import imagezmq

from MTEMode import MTEMode

CROP_SIZE_HOR = 1/3
CROP_SIZE_VER = 1/3

class MTE:
    def __init__(self):
        print("Launching server")
        self.image_hub = imagezmq.ImageHub()
        # self.image_hub = imagezmq.ImageHub(open_port='tcp://192.168.43.39:5555')

        # Prelearning data
        # self.latest_pov_id = -1
        # self.kp_ref = []
        # self.des_ref = []
        # self.ref = []
        self.sift = cv2.xfeatures2d.SIFT_create()

    def listen_images(self):
        while True:  # show streamed images until Ctrl-C
            msg, image = self.image_hub.recv_image()

            data = json.loads(msg)

            # print(data)
            # cv2.imshow(msg, image)

            # key = cv2.waitKey(1)
            # if key == ord("s"):
            #     cv2.imwrite("ref.jpg", image)

            ret_data = {}

            id_pov = data["id_pov"]
            mode = MTEMode(data["mode"])
            if mode == MTEMode.PRELEARNING:
                print("MODE prelearning")
                nb_kp = self.prelearning(id_pov, image)
                # save_ref = "save_ref" in data and data["save_ref"]
                # ret_data["prelearning_pts"] = self.get_rectangle(0, image, force_new_ref=save_ref)
                ret_data["prelearning"] = {
                    "nb_kp": nb_kp
                }
            elif mode == MTEMode.LEARNING:
                print("MODE learning")
                self.learning()
            elif mode == MTEMode.RECOGNITION:
                print("MODE recognition")
                self.recognition()
            # elif mode == MTEMode.FRAMING:
            else:
                print("MODE framing")
                self.framing()

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
        print("Matches found: %d/%d" % (len(goodMatches), MIN_MATCH_COUNT))

        if len(goodMatches) > MIN_MATCH_COUNT:
            dst_pts = np.float32([kp_img[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
            src_pts = np.float32([self.kp_ref[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w = self.ref.shape[:2]
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # img2 = cv2.polylines(img, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

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

    def prelearning(self, id_pov, image):
        # Renvoyer le nombre d'amers sur l'image envoyée
        img = self.crop_image(image)
        kp, _ = self.sift.detectAndCompute(img, None)
        return len(kp)
    
    def learning(self):
        # Enregistrement de l'image de référence en 640 pour SIFT + VC léger et 4K pour VCE
        pass
    
    def recognition(self):
        # Récupération d'une image, SIFT puis si validé VC léger avec mires auto. Si tout ok, envoi image 4K à VCE.
        pass

    def framing(self):
        # Recadrage avec SIFT et renvoi de l'image
        pass

    def crop_image(self, image):
        h, w = image.shape[:2]
        croped = image[int(h*CROP_SIZE_VER/2): int(h*(1-CROP_SIZE_VER/2)), \
            int(w*CROP_SIZE_HOR/2): int(w*(1-CROP_SIZE_HOR/2))]

        return croped

if __name__ == "__main__":
    mte = MTE()
    mte.listen_images()
