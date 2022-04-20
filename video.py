import numpy as np
import cv2
import calibrate_camera
import bird_view
import lanes
import rad
import sobel

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



cap = cv2.VideoCapture('challenge_video.mp4')
# cap = cv2.VideoCapture('project_video.mp4')

video_file = 'output.mp4'
frame_size = (1280, 720)
fps = 40
out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'MP4V'), fps, frame_size)

# Loop until the end of the video
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, frame_size, fx = 0, fy = 0,
                         interpolation = cv2.INTER_CUBIC)
 
        # Display the resulting frame
        # cv2.imshow('Frame', frame)
        result = pipeline(frame, mtx, dist)
        cv2.imshow('image', result)
        # result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        out.write(result)

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