import numpy as np

def measure_curvature_real(binary_warped, left_fitx, right_fitx, ploty):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension may need to be 680 instead of 700
    
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    #calculating offset
    image_center = 1280 / 2
    nonzero = binary_warped[binary_warped.shape[0]//2:,:].nonzero()
    lanes_ceneter = np.sum(nonzero[1]) / len(nonzero[1])
    offset = abs(image_center - lanes_ceneter)
    # print(len(nonzero[1])) # for debugging
    real_offset = xm_per_pix * offset
    
    return left_curverad, right_curverad, real_offset