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
def calibrate(x, y, images_folder):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/*.jpg')
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    img = cv2.imread(images[0])
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    return (mtx, dist)

"""
un-distorts the given image and stores it in the given folder
image: source image
mtx: number of objects points in the y direction
dist: distortion coefficients
id: index to be given to the image as its name
out_folder: dump folder for output
:return void
"""
def undistort(image, mtx, dist, out_folder, id):
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    cv2.imwrite(out_folder + str(id) + '.jpg', dst)

