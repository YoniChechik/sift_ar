# ======= imports
pass

# ======= constants
pass

# === template image keypoint and descriptors
pass

# ===== video input, output and metadata
pass

# ========== run on all frames
while True:
    # ====== find keypoints matches of frame and template
    # we saw this in the SIFT notebook
    pass

    # ======== find homography
    # also in SIFT notebook
    pass

    # ++++++++ take subset of keypoints that obey homography (both frame and reference)
    # this is at most 3 lines- 2 of which are really the same
    # HINT: the function from above should give you this almost completely
    pass

    # ++++++++ solve PnP to get cam pose (r_vec and t_vec)
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
    #
    # this part is 2 rows
    pass

    # ++++++ draw object with r_vec and t_vec on top of rgb frame
    # We saw how to draw cubes in camera calibration. (copy paste)
    # after this works you can replace this with the draw function from the renderer class renderer.draw() (1 line)
    pass

    # =========== plot and save frame
    pass

# ======== end all
pass
