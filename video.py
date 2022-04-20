import numpy as np
import cv2
import calibrate_camera
import bird_view
import lanes
import rad
import sobel
import sys

ksize = 3
mtx, dist = calibrate_camera.calibrate(9, 6, 'camera_cal/*.jpg')
prev_out_img, prev_left_fitx, prev_right_fitx, prev_ploty = (None, None, None, None)

def pipeline(frame, mtx, dist):
    global prev_out_img, prev_left_fitx, prev_right_fitx, prev_ploty

    # phase 1
    undistored_image = calibrate_camera.undistort(frame, mtx, dist)
    # image_rgb = cv2.cvtColor(undistored_image, cv2.COLOR_BGR2RGB)

    # phase 2
    combined_soble = sobel.get_binary(undistored_image, ksize)
    # return combined_soble

    # phase 3
    binary_warped, matrix, matrix_inv = bird_view.get_bird_view(combined_soble)
    
    #phase 4
    out_img, left_fitx, right_fitx, ploty = (None, None, None, None)
    try:
        out_img, left_fitx, right_fitx, ploty = lanes.fit_polynomial(binary_warped, 10, 90, 50)
        prev_out_img, prev_left_fitx, prev_right_fitx, prev_ploty = out_img, left_fitx, right_fitx, ploty
    except:
        out_img, left_fitx, right_fitx, ploty = prev_out_img, prev_left_fitx, prev_right_fitx, prev_ploty
    
    # phase 5
    result = lanes.draw_path(binary_warped, left_fitx, right_fitx, ploty, matrix_inv, frame)
    # calculating curvature and center offset
    left_curverad, right_curverad, real_offset = rad.measure_curvature_real(binary_warped, left_fitx, right_fitx, ploty)
    curve_info = "radius of curvature ({} Km, {} Km)".format(str(round(left_curverad/1000, 2)), 
                                                           str(round(right_curverad/1000, 2)))
    
    center_info = "offset from center  = {} m".format(str(round(real_offset, 2)))
    
    detailed = cv2.putText(result, curve_info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0 , 0), 2, cv2.LINE_AA)
    detailed = cv2.putText(detailed, center_info, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0 , 0), 2, cv2.LINE_AA)
    return result

def debug_pipeline(frame, mtx, dist):
    global prev_out_img, prev_left_fitx, prev_right_fitx, prev_ploty
    undistored_image = calibrate_camera.undistort(frame, mtx, dist)
    combined_soble = sobel.get_binary(undistored_image, ksize)
    binary_warped, matrix, matrix_inv = bird_view.get_bird_view(combined_soble)
    out_img, left_fitx, right_fitx, ploty = (None, None, None, None)
    try:
        out_img, left_fitx, right_fitx, ploty = lanes.fit_polynomial(binary_warped, 10, 90, 50)
        # print("out_img", out_img.shape)
        prev_out_img, prev_left_fitx, prev_right_fitx, prev_ploty = out_img, left_fitx, right_fitx, ploty
    except:
        out_img, left_fitx, right_fitx, ploty = prev_out_img, prev_left_fitx, prev_right_fitx, prev_ploty
    
    result = lanes.draw_path(binary_warped, left_fitx, right_fitx, ploty, matrix_inv, frame)
    left_curverad, right_curverad, real_offset = rad.measure_curvature_real(binary_warped, left_fitx, right_fitx, ploty)
    curve_info = "radius of curvature ({} Km, {} Km)".format(str(round(left_curverad/1000, 2)), 
                                                           str(round(right_curverad/1000, 2)))
    center_info = "offset from center  = {} m".format(str(round(real_offset, 2)))
    detailed = cv2.putText(result, curve_info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0 , 0), 2, cv2.LINE_AA)
    detailed = cv2.putText(detailed, center_info, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0 , 0), 2, cv2.LINE_AA)
    combined_soble = np.dstack((combined_soble, combined_soble, combined_soble))
    binary_warped = np.dstack((binary_warped, binary_warped, binary_warped))
    return undistored_image, combined_soble, binary_warped, out_img, detailed

def get_debug_images(frame, mtx, dist):
    undistored_image, combined_soble, binary_warped, out_img, detailed = debug_pipeline(frame, mtx, dist)
    combined_soble = cv2.resize(combined_soble, (0, 0), None, .5, .5)
    undistored_image = cv2.resize(undistored_image, (0, 0), None, .5, .5)
    binary_warped = cv2.resize(binary_warped, (0, 0), None, .25, .5)
    out_img = cv2.resize(out_img, (0, 0), None, .25, .5)
    detailed = cv2.resize(detailed, (0, 0), None, .25, .5)
    frame =  cv2.resize(frame, (0, 0), None, .25, .5)
    numpy_horz1 = np.hstack((undistored_image, combined_soble*255)) # x * 2
    numpy_horz2 = np.hstack((frame, binary_warped*255, out_img, detailed))
    numpy_ver = np.vstack((numpy_horz1, numpy_horz2))
    return numpy_ver



mode = sys.argv[3]
source = sys.argv[1]
destination = sys.argv[2]
cap = cv2.VideoCapture(source)
# cap = cv2.VideoCapture('project_video.mp4')

frame_size = (1280, 720)
fps = 40
out = cv2.VideoWriter(destination, cv2.VideoWriter_fourcc(*'MP4V'), fps, frame_size)

index = 0
# Loop until the end of the video
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, frame_size, fx = 0, fy = 0,
                         interpolation = cv2.INTER_CUBIC)
 
        # Display the resulting frame
        # cv2.imshow('Frame', frame)
        if(mode == "--production"):
            result = pipeline(frame, mtx, dist)
            out.write(result)
            # cv2.imshow('Frame', result)
        elif(mode == "--debugging"):
            result = get_debug_images(frame, mtx, dist)
            
            # cv2.imshow('Frame', result)
            
            cv2.imwrite('output_images/dump/{}.jpg'.format(index), result)
            new_result = cv2.imread('output_images/dump/{}.jpg'.format(index))
            #cv2.imshow('Frame', new_result)
            out.write(new_result)
            index +=1
        else:
            break

        # define q as the exit button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# release the video capture object
cap.release()
out.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()

print("done")