import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

"""
calculates the transform matrix and un-distortion coefficients
x: number of object points in the x direction
y: number of objects points in the y direction
image_folder: folder containing chess board images for calibration

:return mtx: transformation matrix
        dist: distortion coefficients
"""
def calibrate(x, y, source_images):
    objp = np.zeros((y * x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    object_points = []  # 3d points in real world space
    image_points = []  # 2d points in image plane.

    # Make a list of calibration images paths
    images_location = glob.glob(source_images)

    for index, figure_name in enumerate(images_location):
        image = cv2.imread(figure_name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # because read using cv2.imread
        # Find the chessboard corners
        success, corners = cv2.findChessboardCorners(gray, (x, y), None)
        # If found, add object points, image points
        if success:
            object_points.append(objp)
            image_points.append(corners)

            # code for debugging
            # cv2.drawChessboardCorners(image, (x, y), corners, ret)
            # write_name = 'output_images/drawn_corners/corners_found' + str(index) + '.jpg'
            # cv2.imwrite(write_name, image)

    # choose random image to un-distort
    image = cv2.imread(images_location[0])
    image_size = (image.shape[1], image.shape[0])

    # Do camera calibration given object points and image points
    success, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

    return mtx, dist

"""
un-distorts the given image and stores it in the given folder
image: source image
mtx: number of objects points in the y direction
dist: distortion coefficients
destination: image name.jpg and destination
:return void
"""
def undistort(image, mtx, dist, destination):
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    cv2.imwrite(destination, dst)

