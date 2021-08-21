import cv2
import numpy as np


is_downsample = True

# === ref image
ref_im = cv2.imread("louvre.jpg")
ref_im_rgb = cv2.cvtColor(ref_im, cv2.COLOR_BGR2RGB)
ref_im_gray = cv2.cvtColor(ref_im, cv2.COLOR_BGR2GRAY)


mona = cv2.imread("mona_lisa.jpg")
mona_rgb = cv2.cvtColor(mona, cv2.COLOR_BGR2RGB)
mona_rgb_resize = cv2.resize(mona_rgb, (ref_im.shape[1], ref_im.shape[0]))


feature_extractor = cv2.SIFT_create()

kp_l, desc_l = feature_extractor.detectAndCompute(ref_im_gray, None)


# ===== video input
cap = cv2.VideoCapture("vid.mp4")
fps = np.round(cap.get(cv2.CAP_PROP_FPS))
width = int(np.round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
height = int(np.round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# ==== video write
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, fps, (width, height))


# ========== run on all frames
frame_num = -1
while True:
    flag, frame = cap.read()
    if not flag:
        break

    frame_num += 1
    print("frame num " + str(frame_num))

    if is_downsample:
        if not frame_num % 10 == 0:
            continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find the keypoints and descriptors with chosen feature_extractor
    kp_r, desc_r = feature_extractor.detectAndCompute(frame_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_l, desc_r, k=2)

    # Apply ratio test
    good_and_second_good_match_list = []
    for m in matches:
        if m[0].distance / m[1].distance < 0.5:
            good_and_second_good_match_list.append(m)
    good_match_arr = np.asarray(good_and_second_good_match_list)[:, 0]

    if good_match_arr.size == 0:
        print("skip frame " + str(frame_num))
        continue

    good_kp_l = np.array([kp_l[m.queryIdx].pt for m in good_match_arr])
    good_kp_r = np.array([kp_r[m.trainIdx].pt for m in good_match_arr])

    H, masked = cv2.findHomography(good_kp_l, good_kp_r, cv2.RANSAC, 5.0)
    if not isinstance(H, np.ndarray):
        print(f"skip frame {frame_num}")
        continue

    # ======== do perspective warping
    mask_warped = cv2.warpPerspective(
        np.ones(ref_im.shape, dtype=np.uint8), H, (frame_rgb.shape[1], frame_rgb.shape[0])
    )
    mask_warped_bin = mask_warped > 0
    im_warped = cv2.warpPerspective(mona_rgb_resize, H, (frame_rgb.shape[1], frame_rgb.shape[0]))

    frame_rgb[mask_warped_bin] = im_warped[mask_warped_bin]

    # =========== output
    cv2.imshow("frame", frame_rgb)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    out.write(frame_rgb)


# ======== end all
cap.release()
cv2.destroyAllWindows()
print("====== finished ======")
