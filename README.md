## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./examples/example_output.jpg)

### Get Started
#### 1. Make Sure to Download All Dependencies
#### 2. Download The Repo
#### 3. Run any of the following commands in the commmand line: 
- full production video `./run.sh <input_file_path> <output_file_path> --production`
![AnimationGeneral](https://user-images.githubusercontent.com/49312818/170526459-cb87d8b5-5443-452b-b155-368ae9496757.gif)

- production video for the lane detection `./run.sh <input_file_path> <output_file_path> --production --lanes`
- production video for car detection with yolo `./run.sh <input_file_path> <output_file_path> --production --yolo`
- production video for car detection with HOG + SVM `./run.sh <input_file_path> <output_file_path> --production --hog`
- debugging video for lane detection `./run.sh <input_file_path> <output_file_path> --debugging --lanes`
- debugging video for HOG + SVM `./run.sh <input_file_path> <output_file_path> --debugging --hog`

## Installation

### Dependencies
**"yolov3.weights" file should be added in "model_data" folder**

```
pip3 install numpy
```
```
pip3 install matplotlib
```
```
pip3 install glob2
```
```
pip3 install opencv-python
```
```
pip3 install moviepy
```
```
pip install -U scikit-learn
```
```
pip install -U scikit-image
```
```
pip install pickle4
```
---

## Methodology

### Lane Detection
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Car Detection (YOLO)
I used (YOLO) "You Only Look Once" because it is a popular algorithm because it achieves high accuracy while also being able to run in real time. This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

### Car Detection (HOG)
- crop the image to the region of interest (ROI) "the lower half of the image"
- extract the hod features for the whole region
- decide a suitable window size and slide it along the image
- store the corners of these windows that found a car
- draw a heat map of these windows
- taking a threshold of the heat map to get a better measurement
- average the windows locations along many frame to reduce the noise draw the results

## folder Structure
```
Perception-Stack
├── camera_cal/         chessboard images used to calibrate the camera
├── examples/            examples of the expected output from the starter files
├── model_data/          includes data of the yolo model
├── output_images/       output images of the different stages of the pipeline
├── test_images/         road images used for testing
├── bird_view.py        contains fucntion to generate the bird prespective view
├── calibrate_camera/   functions to calibrate the camera
├── lanes/              function for detecting, fitting, and plotting the lanes
├── main.html/          the writeup 
├── main.ipynb/         the main project notebook
├── rad.py              function for calulating offset from center and radius of curvature
├── run.sh              bash script to run the project
├── repo_link.txt       the repo link 
├── README.md           main documentation.
├── SysMonitor.py       used to identify and stop the program.
├── sobel.py            detects the lane lines.
├── HOG.py              extracts the hog features for the svm
├── svc_pickle.p        includes the parameters for the svm
└── video.py            generates production and debugging videos
```
