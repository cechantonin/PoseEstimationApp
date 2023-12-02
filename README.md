# Pose Estimation and Pose Tracking Application

This repository contains the application for the visualization of human pose tracking and estimation. The application can be used to run pose estimation on a video with enabled or disabled pose tracking and save the visualized video along with the results in a JSON file. The JSON files can then be opened in the application and visualized on the video.

It is written in Python 3.10 and uses packages [MMDet](https://github.com/open-mmlab/mmdetection), [MMTrack](https://github.com/open-mmlab/mmtracking) and [MMPose](https://github.com/open-mmlab/mmpose). These packages use PyTorch and CUDA supported graphics card is needed to run the application. The user interface is made using the PyQt5 package.

# Requirements
- Python 3.10
- PyQt5
- PyTorch 1.13.0
- MMDet 2.28.2
- MMTrack 0.14.0
- MMPose 0.29.0

# Installation
For installation check [Installation.md](Installation.md)
