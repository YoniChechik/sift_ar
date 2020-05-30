import cv2
import numpy as np
# import matplotlib.pyplot as plt


is_downsample = False
is_cube = True

K = np.array([[1.53653164e+03, 0.00000000e+00, 1.04318348e+03],
              [0.00000000e+00, 1.54022929e+03, 5.67584058e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coeffs = np.array([0.2897652,  -2.45780323, 0.01189844, 0.02099917, 6.97784126])

# image size in cm
REF_IM_W = 28
REF_IM_H = 19

# === ref image
ref_im = cv2.imread("louvre.jpg")
ref_im_rgb = cv2.cvtColor(ref_im, cv2.COLOR_BGR2RGB)
ref_im_gray = cv2.cvtColor(ref_im, cv2.COLOR_BGR2GRAY)
# plt.imshow(im)
# plt.show()

feature_extractor = cv2.xfeatures2d.SIFT_create()

kp_ref, desc_ref = feature_extractor.detectAndCompute(ref_im_gray, None)
kp_ref_XY = [[x.pt[0]/ref_im_gray.shape[1]*REF_IM_W,
              x.pt[1]/ref_im_gray.shape[0]*REF_IM_H,
              0] for x in kp_ref]


# ===== video input
cap = cv2.VideoCapture('vid.mp4')
fps = np.round(cap.get(cv2.CAP_PROP_FPS))
width = int(np.round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
height = int(np.round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# ==== video write
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

# ================= 3D object
if is_cube:

    class basic_cube_render():
        def __init__(self):
            self.objectPoints = 10*np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, -1],
                                             [0, 1, -1], [1, 1, -1], [1, 0, -1]], dtype=float)

        def draw(self, img, rvec, tvec):
            imgpts = cv2.projectPoints(self.objectPoints, rvec, tvec, K, dist_coeffs)[0]

            imgpts = np.int32(imgpts).reshape(-1, 2)

            # draw ground floor in green
            img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -1)

            # draw pillars in blue color
            for i, j in zip(range(4), range(4, 8)):
                img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

            # draw top layer in red color
            img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

            return img

    render = basic_cube_render()


else:
    import trimesh

    def rot_x(t):
        ct = np.cos(t)
        st = np.sin(t)
        m = np.array([[1, 0, 0],
                      [0, ct, -st],
                      [0, st, ct]])
        return m

    mesh = trimesh.load('drill/drill.obj')
    # normalize bounding box from (0,0,0) to max(30)
    mesh.rezero()  # set th LOWER LEFT (?) as (0,0,0)
    T = np.eye(4)
    T[0:3, 0:3] = 10*np.eye(3)*(1 / np.max(mesh.bounds))
    mesh.apply_transform(T)
    # rotate to make the drill standup
    T = np.eye(4)
    T[0:3, 0:3] = rot_x(np.pi/2)
    mesh.apply_transform(T)

    class mesh_render():

        def __init__(self, mesh):
            import pyrender

            # rotate 180 around x because the Z dir of the reference grid is down
            T = np.eye(4)
            T[0:3, 0:3] = rot_x(np.pi)
            mesh.apply_transform(T)
            # Load the trimesh and put it in a scene
            mesh = pyrender.Mesh.from_trimesh(mesh)
            scene = pyrender.Scene(bg_color=np.array([0, 0, 0, 0]))
            scene.add(mesh)

            # add temp cam
            self.camera = pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], zfar=10000, name="cam")
            light_pose = np.array([
                [1.0, 0,   0,   0.0],
                [0,  1.0, 0.0, 10.0],
                [0.0,  0,   1,   100.0],
                [0.0,  0.0, 0.0, 1.0],
            ])
            self.cam_node = scene.add(self.camera, pose=light_pose)

            # Set up the light -- a single spot light in z+
            light = pyrender.SpotLight(color=255*np.ones(3), intensity=3000.0,
                                       innerConeAngle=np.pi/16.0)
            scene.add(light, pose=light_pose)

            self.scene = scene
            self.r = pyrender.OffscreenRenderer(width, height)
            # add the A flag for the masking
            self.flag = pyrender.constants.RenderFlags.RGBA

        def draw(self, img, rvec, tvec):
            # ===== update cam pose
            camera_pose = np.eye(4)
            res_R, _ = cv2.Rodrigues(rvec)

            # opengl transformation
            # https://stackoverflow.com/a/18643735/4879610
            camera_pose[0:3, 0:3] = res_R.T
            camera_pose[0:3, 3] = (-res_R.T@tvec).flatten()
            # 180 about x
            camera_pose = camera_pose@np.array([[1, 0, 0, 0],
                                                [0, -1, 0, 0],
                                                [0, 0, -1, 0],
                                                [0, 0, 0, 1]])

            self.scene.set_pose(self.cam_node, camera_pose)

            # ====== Render the scene
            color, depth = self.r.render(self.scene, flags=self.flag)
            img[color[:, :, 3] != 0] = color[:, :, 0:3][color[:, :, 3] != 0]
            return img

    render = mesh_render(mesh)


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

    # plt.imshow(frame)
    # plt.show()

    # find the keypoints and descriptors with chosen feature_extractor
    kp_frame, desc_frame = feature_extractor.detectAndCompute(frame_gray, None)

    # test = cv2.drawKeypoints(ref_im, kp_ref, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.figure()
    # plt.imshow(frame_rgb)
    # plt.title("keypoints")
    # plt.show()

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_ref, desc_frame, k=2)

    # Apply ratio test
    good_match = []
    for m in matches:
        if m[0].distance/m[1].distance < 0.5:
            good_match.append(m)
    good_match_arr = np.asarray(good_match)

    # show only 30 matches
    # im_matches = cv2.drawMatchesKnn(im_gray, kp_l, frame_gray, kp_r,
    #                                 good_match[0:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.figure(figsize=(20, 20))
    # plt.imshow(im_matches)
    # plt.title("keypoints matches")
    # plt.show()

    if good_match_arr.size == 0:
        print("skip frame "+str(frame_num))
        continue

    good_kp_ref = np.array([kp_ref[m.queryIdx].pt for m in good_match_arr[:, 0]]).reshape(-1, 1, 2)
    good_kp_ref_XY = [kp_ref_XY[m.queryIdx] for m in good_match_arr[:, 0]]

    good_kp_frame = np.array([kp_frame[m.trainIdx].pt for m in good_match_arr[:, 0]]).reshape(-1, 1, 2)

    # ===== find the points that obbay homography
    H, masked = cv2.findHomography(good_kp_ref, good_kp_frame, cv2.RANSAC, 5.0)
    if not isinstance(H, np.ndarray):
        print("skip frame "+str(frame_num))
        continue

    best_kp_ref_XY = np.array(good_kp_ref_XY)[np.array(masked).flatten() > 0, :]
    best_kp_r = np.array(good_kp_frame.reshape(good_kp_frame.shape[0], 2))[np.array(masked).flatten() > 0, :]

    # ========= solve PnP to get cam pos
    res, rvec, tvec = cv2.solvePnP(best_kp_ref_XY[:, :, np.newaxis], best_kp_r[:, :, np.newaxis], K, dist_coeffs)
    # res_R,_ = cv2.Rodrigues(rvec)
    # x = np.array([[0,0,0]]).T
    # x_cam2_world = K@(res_R@x+tvec)
    # x2_norm = x_cam2_world[:2,:]/x_cam2_world[2,:]
    # plt.figure()
    # plt.imshow(frame_rgb)
    # plt.plot(x2_norm[0, 0], x2_norm[1, 0], '*w')
    # plt.show()

    # ======== draw proj res
    drawn_image = render.draw(frame_rgb, rvec, tvec)
    # plt.imshow(drawn_image)
    # plt.show()

    # =========== output
    final_res = cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('frame', final_res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    out.write(final_res)


# ======== end all
cap.release()
cv2.destroyAllWindows()
print("====== finished ======")
