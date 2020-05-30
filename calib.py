import numpy as np
import cv2
import matplotlib.pyplot as plt

square_size = 2.5  # [cm]
pattern_size = (9, 6)

figsize = (20, 20)


pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

obj_points = []
img_points = []


plt.figure(figsize=figsize)


# ===== video input
cap = cv2.VideoCapture('calib_vid_hor.mp4')
fps = np.round(cap.get(cv2.CAP_PROP_FPS))
w = int(np.round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
h = int(np.round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


frame_num = -1
i = -1
res_imgs = []
while True:
    flag, imgBGR = cap.read()
    if not flag:
        break

    frame_num += 1
    if frame_num % fps != 0:
        continue
    i += 1

    res_imgs.append(imgBGR)
    print('processing %s... ' % i)

    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)

    assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
    found, corners = cv2.findChessboardCorners(img, pattern_size)
    # # if you want to better improve the accuracy... cv2.findChessboardCorners already uses cv2.cornerSubPix
    # if found:
    #     term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    #     cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

    if not found:
        print('chessboard not found')
        continue

    if i < 12:
        img_w_corners = cv2.drawChessboardCorners(imgRGB, pattern_size, corners, found)
        plt.subplot(4, 3, i+1)
        plt.imshow(img_w_corners)

    print('           %s... OK' % i)
    img_points.append(corners.reshape(-1, 2))
    obj_points.append(pattern_points)

cap.release()
cv2.destroyAllWindows()

plt.show()

# calculate camera distortion
rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

print("\nRMS:", rms)
print("camera matrix:\n", camera_matrix)
print("distortion coefficients: ", dist_coefs.ravel())


# =========== test results

objectPoints = 3*square_size*np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                                       [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]])


def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -1)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


plt.figure(figsize=figsize)
for i, imgBGR in enumerate(res_imgs):

    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

    dst = cv2.undistort(imgRGB, camera_matrix, dist_coefs)

    imgpts = cv2.projectPoints(	objectPoints, _rvecs[i], _tvecs[i], camera_matrix, dist_coefs)[0]
    drawn_image = draw(dst, imgpts)

    if i < 12:
        plt.subplot(4, 3, i+1)
        plt.imshow(drawn_image)

plt.show()
