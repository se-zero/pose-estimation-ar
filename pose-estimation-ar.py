import numpy as np
import cv2 as cv

# The given video and calibration data
video_file = 'data/chessboard.mp4'
K = np.array([
    [911.06897706, 0, 959.39682216],
    [0, 912.93482348, 536.39064935],
    [0, 0, 1]
])
dist_coeff = np.array([-0.0104789, 0.01314287, -0.00267109, 0.00416095, -0.00936107])
board_pattern = (8, 6)
board_cellsize = 0.029
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

# Prepare a 3D box for simple AR
a_face_front = board_cellsize * np.array([
    [3, 5, 0], 
    [3, 4, 0], 
    [3, 3.5, -2], 
    [3, 3, 0],
    [3, 2, 0], 
    [3, 3.5, -3] 
    ])
a_face_back = board_cellsize * np.array([
    [2, 5, 0], 
    [2, 4, 0], 
    [2, 3.5, -2], 
    [2, 3, 0],
    [2, 2, 0], 
    [2, 3.5, -3] 
    ])

plus_face_front = board_cellsize * np.array([
    [3, 2, -1], 
    [3, 1, -1], 
    [3, 1, 0], 
    [3, 0, 0],
    [3, 0, -1], 
    [3, -1, -1],
    [3, -1, -2], 
    [3, 0, -2], 
    [3, 0, -3], 
    [3, 1, -3],
    [3, 1, -2], 
    [3, 2, -2]
    ])
plus_face_back = board_cellsize * np.array([
    [2, 2, -1], 
    [2, 1, -1], 
    [2, 1, 0], 
    [2, 0, 0],
    [2, 0, -1], 
    [2, -1, -1],
    [2, -1, -2], 
    [2, 0, -2], 
    [2, 0, -3], 
    [2, 1, -3],
    [2, 1, -2], 
    [2, 2, -2] 
    ])

# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Draw the box on the image
        a_proj_front, _ = cv.projectPoints(a_face_front, rvec, tvec, K, dist_coeff)
        a_proj_back, _ = cv.projectPoints(a_face_back, rvec, tvec, K, dist_coeff)
        cv.polylines(img, [np.int32(a_proj_front)], True, (255, 0, 0), 2)
        cv.polylines(img, [np.int32(a_proj_back)], True, (0, 0, 255), 2)
        for b, t in zip(a_proj_front, a_proj_back):
            cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)
            
         
        plus_proj_front, _ = cv.projectPoints(plus_face_front, rvec, tvec, K, dist_coeff)
        plus_proj_back, _ = cv.projectPoints(plus_face_back, rvec, tvec, K, dist_coeff)
        cv.polylines(img, [np.int32(plus_proj_front)], True, (255, 100, 0), 2)
        cv.polylines(img, [np.int32(plus_proj_back)], True, (0, 100, 255), 2)
        for b, t in zip(plus_proj_front, plus_proj_back):
            cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (100, 255, 100), 2)

        # Print the camera position
        R, _ = cv.Rodrigues(rvec) # Alternative) `scipy.spatial.transform.Rotation`
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Chessboard)', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27: # ESC
        break

video.release()
cv.destroyAllWindows()