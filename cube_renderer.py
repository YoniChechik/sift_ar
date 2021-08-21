import numpy as np
import cv2


class BasicCubeRenderer:
    def __init__(self, K, dist_coeffs):
        self.objectPoints = 10 * np.array(
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]], dtype=float
        )
        self.K = K
        self.dist_coeffs = dist_coeffs

    def draw(self, img, rvec, tvec):
        imgpts = cv2.projectPoints(self.objectPoints, rvec, tvec, self.K, self.dist_coeffs)[0]

        imgpts = np.int32(imgpts).reshape(-1, 2)

        # draw ground floor in green
        img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -1)

        # draw pillars in blue color
        for i, j in zip(range(4), range(4, 8)):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

        # draw top layer in red color
        img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

        return img
