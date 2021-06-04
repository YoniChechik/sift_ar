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