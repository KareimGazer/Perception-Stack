## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./examples/example_output.jpg)

In this project, I detect the lane lines using ...

Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

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
Perception
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
