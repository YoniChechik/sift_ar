# ======= imports
import cv2
import numpy as np
import mesh_renderer
import cube_renderer

# ======= constants
is_downsample = True

K = np.array(
    [
        [1.53653164e03, 0.00000000e00, 1.04318348e03],
        [0.00000000e00, 1.54022929e03, 5.67584058e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)
dist_coeffs = np.array([0.2897652, -2.45780323, 0.01189844, 0.02099917, 6.97784126])

# ref image size in cm
TEMPLATE_IM_W = 28
TEMPLATE_IM_H = 19

template_image_path = "louvre.jpg"
video_input = "vid.mp4"


# === ref image keypoint and descriptors
template_im = cv2.imread(template_image_path)
template_im_gray = cv2.cvtColor(template_im, cv2.COLOR_BGR2GRAY)

feature_extractor = cv2.SIFT_create()

kp_template, desc_template = feature_extractor.detectAndCompute(template_im_gray, None)

# ===== video input and metadata
cap = cv2.VideoCapture(video_input)
fps = np.round(cap.get(cv2.CAP_PROP_FPS))
width = int(np.round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
height = int(np.round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# ====== build renderer
renderer_obj = cube_renderer.BasicCubeRenderer(K,dist_coeffs)
renderer_obj = mesh_renderer.MeshRenderer(K, width, height, "drill/drill.obj")


# ==== video write init
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

    # ====== find keypoints matches of frame and template
    kp_frame, desc_frame = feature_extractor.detectAndCompute(frame_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_template, desc_frame, k=2)

    # Apply ratio test
    good_and_second_good_match_list = []
    for m in matches:
        if m[0].distance / m[1].distance < 0.5:
            good_and_second_good_match_list.append(m)
    good_match_arr = np.asarray(good_and_second_good_match_list)[:,0]

    if good_match_arr.size == 0:
        print("skip frame " + str(frame_num))
        continue

    good_kp_template = np.array([kp_template[m.queryIdx].pt for m in good_match_arr])
    good_kp_frame = np.array([kp_frame[m.trainIdx].pt for m in good_match_arr])

    # ======== find homography
    H, masked = cv2.findHomography(good_kp_template, good_kp_frame, cv2.RANSAC, 5.0)
    if not isinstance(H, np.ndarray):
        print("skip frame " + str(frame_num))
        continue

    # ===== take subset of keypoints that obey homography (both frame and reference)
    good_homography_mask = (np.array(masked).flatten() > 0)
    best_kp_template = np.array(good_kp_template)[good_homography_mask, :]
    best_kp_frame = np.array(good_kp_frame.reshape(good_kp_frame.shape[0], 2))[good_homography_mask, :]

    # ========= solve PnP to get cam pose (r_vec and t_vec)
    # `cv2.solvePnP` is a function that receives:
    # - xyz of the template in centimeter in camera world (x,3)
    # - uv coordinates (x,2) of frame that corresponds to the xyz triplets
    # - camera K
    # - camera dist_coeffs
    # and outputs the camera pose (r_vec and t_vec) such that the uv is aligned with the xyz.
    #
    # NOTICE: the first input to `cv2.solvePnP` is (x,3) vector of xyz in centimeter- but we have the template keypoints in uv
    # because they are all on the same plane we can assume z=0 and simply rescale each keypoint to the ACTUAL WORLD SIZE IN CM.
    # For this we just need the template width and height in cm.
    best_kp_template_xyz_cm = np.array([[x[0] / template_im_gray.shape[1] * TEMPLATE_IM_W, x[1] / template_im_gray.shape[0] * TEMPLATE_IM_H, 0] for x in best_kp_template])
    res, rvec, tvec = cv2.solvePnP(best_kp_template_xyz_cm, best_kp_frame, K, dist_coeffs)

    # ======== draw object with r_vec and t_vec on top of rgb frame
    drawn_image = renderer_obj.draw(frame_rgb, rvec, tvec)

    # =========== plot and save frame
    final_res = cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("frame", final_res)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    out.write(final_res)

# ======== end all
cap.release()
cv2.destroyAllWindows()
print("====== finished ======")
