import numpy as np
import cv2
import calibrate_camera
import bird_view
import lanes
import rad
import sobel
import sys
from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from lesson_functions import *
from functools import reduce


ksize = 3
mtx, dist = calibrate_camera.calibrate(9, 6, 'camera_cal/*.jpg')
prev_out_img, prev_left_fitx, prev_right_fitx, prev_ploty = (None, None, None, None)

weights_path = 'model_data/yolov3.weights'
confg_path = 'model_data/yolov3.cfg'
labels_path = 'model_data/coco.names'
labels = open(labels_path).read().strip().split('\n')
net = cv2.dnn.readNetFromDarknet(confg_path, weights_path)
out_layer_name = net.getUnconnectedOutLayersNames()

# get attributes of our svc object
svc = None
X_scaler = None
orient = None
pix_per_cell = None
cell_per_block = None

heat_history = []
ystart = 400 # 330 650
ystop = 656
scale = 1.5 # 
move_pix = 1 #  4 cells_per_step in the lesson
frames_to_remember = 2


def detect_lanes(frame):
    global prev_out_img, prev_left_fitx, prev_right_fitx, prev_ploty, mtx, dist, ksize
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    undistored_image = calibrate_camera.undistort(frame, mtx, dist)
    # image_rgb = cv2.cvtColor(undistored_image, cv2.COLOR_BGR2RGB)
    combined_soble = sobel.get_binary(undistored_image, ksize)
    # return combined_soble

    binary_warped, matrix, matrix_inv = bird_view.get_bird_view(combined_soble)
    
    out_img, left_fitx, right_fitx, ploty = (None, None, None, None)
    try:
        out_img, left_fitx, right_fitx, ploty = lanes.fit_polynomial(binary_warped, 10, 90, 50)
        prev_out_img, prev_left_fitx, prev_right_fitx, prev_ploty = out_img, left_fitx, right_fitx, ploty
    except:
        out_img, left_fitx, right_fitx, ploty = prev_out_img, prev_left_fitx, prev_right_fitx, prev_ploty
    
    result = lanes.draw_path(binary_warped, left_fitx, right_fitx, ploty, matrix_inv, frame)
    
    # calculating curvature and center offset
    left_curverad, right_curverad, real_offset = rad.measure_curvature_real(binary_warped, left_fitx, right_fitx, ploty)
    curve_info = "radius of curvature ({} Km, {} Km)".format(str(round(left_curverad/1000, 2)), 
                                                           str(round(right_curverad/1000, 2)))
    
    center_info = "offset from center  = {} m".format(str(round(real_offset, 2)))
    
    detailed = cv2.putText(result, curve_info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0 , 0), 2, cv2.LINE_AA)
    detailed = cv2.putText(detailed, center_info, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0 , 0), 2, cv2.LINE_AA)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


"""
returns group of images each represents a step in the pipeline
"""
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


"""
resizes the debug pipeline images so it can fit into a single frame
"""
def get_debug_image(frame):
    global mtx, dist
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    undistored_image, combined_soble, binary_warped, out_img, detailed = debug_pipeline(frame, mtx, dist)
    
    undistored_image = cv2.cvtColor(undistored_image, cv2.COLOR_BGR2RGB)
    detailed = cv2.cvtColor(detailed, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
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


def detect_cars_yolo(frame):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (256, 256), crop=False, swapRB=False) # check RB, 1/255.0
    net.setInput(blob)
    net_out = net.forward(out_layer_name)
    frame_height, frame_width = frame.shape[:2]
    boxes = []
    confidences = []
    classIDs = []
    for output in net_out:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if(confidence > 0.9):
                box = detection[:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                bx, by, bw, bh = box.astype('int')
                x, y = int(bx - (bw/2)), int(by - bh/2)
                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.6)
    if not len(idxs):
        return frame
    for i in idxs.flatten():
        x, y = [boxes[i][0], boxes[i][1]]
        w, h = [boxes[i][2], boxes[i][3]]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255),2)
        cv2.putText(frame, '{}:{:.2f}'.format(labels[classIDs[i]], confidences[i]), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 139,  139), 2)
    return frame


def full_perception(frame):
    lanes_detected = detect_lanes(frame)
    cars_detected = detect_cars_yolo(lanes_detected)
    return cars_detected


def detect_cars_hog(image):
    threshold = 5
    global heat_history, move_pix, ystart, ystop, scale, svc 
    global frames_to_remember, X_scaler, orient, pix_per_cell, cell_per_block
    box_list = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, move_pix)
    
    heat = np.zeros_like(image[:,:,0]).astype(float)
    heat = add_heat(heat, box_list)
    if len(heat_history) >= frames_to_remember:
            heat_history = heat_history[1:]
    heat_history.append(heat)
    
    heat = reduce(lambda h, acc: h + acc, heat_history)
    heat = apply_threshold(heat, threshold)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    
    return draw_img

def debug_image_hog(frame):
    threshold = 5
    global heat_history, move_pix, ystart, ystop, scale, svc 
    global frames_to_remember, X_scaler, orient, pix_per_cell, cell_per_block
    
    boxes_image, box_list = find_cars_boxes(frame, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, move_pix)
    
    heat = np.zeros_like(frame[:,:,0]).astype(float)
    heat = add_heat(heat, box_list)
    heat = apply_threshold(heat, threshold)
    
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    
    draw_img = draw_labeled_bboxes(np.copy(frame), labels)
    
    frame = cv2.resize(frame, (0, 0), None, .5, .5)
    boxes_image = cv2.resize(boxes_image, (0, 0), None, .5, .5)
    
    heatmap = np.dstack((heatmap, heatmap, heatmap))
    heatmap = cv2.resize(heatmap, (0, 0), None, .5, .5)
    draw_img = cv2.resize(draw_img, (0, 0), None, .5, .5)
    
    numpy_horz1 = np.hstack((frame, boxes_image)) # x * 2
    numpy_horz2 = np.hstack((heatmap*255, draw_img))
    numpy_ver = np.vstack((numpy_horz1, numpy_horz2))
    return numpy_ver


project_video_path = sys.argv[1]
project_video_output = sys.argv[2]
mode = sys.argv[3]
kind = "--general"
if (len(sys.argv) > 4):
    kind = sys.argv[4]

project_video = VideoFileClip(project_video_path)
print("Mode: ", mode, "  type: ", kind)

if(mode == "--production" and kind == "--yolo"):
    out_clip = project_video.fl_image(detect_cars_yolo) 
    out_clip.write_videofile(project_video_output, audio=False)
elif(mode == "--production" and kind == "--hog"):
    import pickle

    file_name = "svc_pickle.p"
    # load a pe-trained svc model from a serialized (pickle) file
    dist_pickle = pickle.load(open(file_name, "rb" ))

    # get attributes of our svc object
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    out_clip = project_video.fl_image(detect_cars_hog) 
    out_clip.write_videofile(project_video_output, audio=False)
elif(mode == "--production" and kind == "--lanes"):
    out_clip = project_video.fl_image(detect_lanes) 
    out_clip.write_videofile(project_video_output, audio=False)
elif(mode == "--production"):
    out_clip = project_video.fl_image(full_perception) 
    out_clip.write_videofile(project_video_output, audio=False)
elif(mode == "--debugging" and kind == "--hog"):
    import pickle

    file_name = "svc_pickle.p"
    # load a pe-trained svc model from a serialized (pickle) file
    dist_pickle = pickle.load(open(file_name, "rb" ))

    # get attributes of our svc object
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    out_clip = project_video.fl_image(debug_image_hog)
    out_clip.write_videofile(project_video_output, audio=False)
elif(mode == "--debugging" and kind == "--lanes"):
    out_clip = project_video.fl_image(get_debug_image)
    out_clip.write_videofile(project_video_output, audio=False)

print("done")
