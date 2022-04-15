import numpy as np
import cv2

"""
applies Sobel x or y, then computes the absolute value of the gradient and applies a threshold.
parameters:
  img: source image
  sobel_kernel: symmetric kernel size
  thresh: range of desired values
"""
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_abs = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    grad_binary = np.zeros_like(scaled_abs)
    grad_binary[(scaled_abs >= thresh_min) & (scaled_abs <= thresh_max)] = 1
    return grad_binary


"""
applies Sobel x and y, then computes the magnitude of the gradient and applies a threshold.
parameters:
  img: source image
  sobel_kernel: symmetric kernel size
  thresh: range of desired values
"""
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude 
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))
    scaled_mag = np.uint8(255*mag/np.max(mag))
    
    thresh_min = mag_thresh[0]
    thresh_max = mag_thresh[1]
    binary_output = np.zeros_like(scaled_mag)
    binary_output[(scaled_mag >= thresh_min) & (scaled_mag <= thresh_max)] = 1
    return binary_output


"""
applies Sobel x and y, then computes the direction of the gradient
and applies a threshold.
parameters:
  img: source image
  sobel_kernel: symmetric kernel size
  thresh: range of desired values
"""
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    arc_tan = np.arctan2(abs_sobely, abs_sobelx)
    
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    binary_output = np.zeros_like(arc_tan)
    binary_output[(arc_tan >= thresh_min) & (arc_tan <= thresh_max)] = 1
    return binary_output

"""
combines both sobel gradients on RGB images and color threshold on s channel in HLS image
"""
def get_binary(image, ksize):
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1)) # 0.7, 1.3 (old values)
    combined_soble = np.zeros_like(dir_binary)
    combined_soble[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_thresh_min = 150
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    combined_binary = np.zeros_like(combined_soble)
    combined_binary[(s_binary == 1) | (combined_soble == 1)] = 1
    return combined_binary