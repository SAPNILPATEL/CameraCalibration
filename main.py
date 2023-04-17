import cv2
import numpy as np
import glob

# Number of corners in the checkerboard
CHECKERBOARD_SIZE = (6, 8)

# Criteria for corner detection
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store 3D points (object points) and 2D points (image points)
obj_points = []
img_points = []

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (5,7,0)
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)

# Load images
images = glob.glob('img1.png')  # Replace with the path to your images

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find corners on the checkerboard
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)

    # If corners are found, add object points and image points
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        cv2.drawChessboardCorners(img, CHECKERBOARD_SIZE, corners, ret)
        obj_points.append(objp)
        img_points.append(corners)

    # Show images with detected corners
    cv2.imshow('Camera Calibration', img)
    cv2.waitKey(500)

# Calibrate camera
ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Print camera calibration parameters
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", distortion_coeffs)

# Close all OpenCV windows
cv2.destroyAllWindows()
