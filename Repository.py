import os
import time
import sqlite3
import cv2
import imutils

from Domain.LearningData import LearningData

IMAGES_FOLDER = "images/"

class Repository:
    def __init__(self):
        self.conn = sqlite3.connect('mte.db')
        self.cursor = self.conn.cursor()

        if not os.path.exists(IMAGES_FOLDER):
            os.makedirs(IMAGES_FOLDER)

    def create_tables(self):
        # Create table
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS `mte_pov` (
            `id` INTEGER PRIMARY KEY,
            `label` TEXT,
            `image_640` TEXT,
            `full_image` TEXT
        );''')
        self.conn.commit()

    def close(self):
        self.conn.close()

    def save_new_pov(self, full_image):
        millis = int(round(time.time() * 1000))

        full_image_name = str(millis) + ".jpg"
        full_image_path = os.path.join(IMAGES_FOLDER, full_image_name)
        cv2.imwrite(full_image_path, full_image)

        image_640 = imutils.resize(full_image, width=640)
        image_640_name = str(millis) + "_640.jpg"
        image_640_path = os.path.join(IMAGES_FOLDER, image_640_name)
        cv2.imwrite(image_640_path, image_640)

        params = (image_640_name, full_image_name)
        self.cursor.execute("INSERT INTO `mte_pov` (`label`, `image_640`, `full_image`) VALUES ('',?,?)", params)
        self.conn.commit()

        return self.cursor.lastrowid

    def get_pov_by_id(self, pov_id):
        params = (pov_id, )
        self.cursor.execute('SELECT * FROM `mte_pov` WHERE `id`=?', params)
        result = self.cursor.fetchone()

        success = not result is None

        if success:
            image_640 = cv2.imread(os.path.join(IMAGES_FOLDER, result[2]))
            full_image = cv2.imread(os.path.join(IMAGES_FOLDER, result[3]))
            data = LearningData(result[0], result[1], image_640, full_image)
        else:
            data = None

        return success, data

if __name__ == "__main__":
    repo = Repository()
    repo.create_tables()

    # Test creating row
    # img = cv2.imread("images/loup-4k.jpg")
    # repo.save_new_pov(img)

    # Test reading
    repo.get_pov_by_id(1)

    repo.close()
