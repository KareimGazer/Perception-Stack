## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./examples/example_output.jpg)


## Installation

### Dependencies
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
---
### Get Started
- download the repo
- to generatge the production video `./run.sh <input_file_path> <output_file_path> --production`
- to generate the debugging video `./run.sh <input_file_path> <output_file_path> --debugging`



The Project Steps
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## folder Structure
```
Perception-Stack
├── camera_cal/         chessboard images used to calibrate the camera
├── examples/            examples of the expected output from the starter files
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
└── video.py            generates production and debugging videos
```
