import numpy as np
import cv2

# pass matplot image
def get_bird_view(image):
    width, height = (image.shape[1], image.shape[0])
    offset = 300
    src_points = np.float32([[575, 460], [710, 460], [210, 690], [1080, 690]])
    dst_points = np.float32([[offset, 0], [width-offset, 0], [offset, height], [width-offset, height]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    matrix_inv = cv2.getPerspectiveTransform(dst_points, src_points)
    out = cv2.warpPerspective(image, matrix, (width, height))
    return out
    