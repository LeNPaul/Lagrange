---
layout: post
title: "Camera Calibration using Zhang's Method"
author: "Paul Vinh Phan"
categories: journal
image: reprojection.gif
tags: [camera,calibration,intrinsic,extrinsic]
---

The gif above shows the reprojection of expected target points (magenta) vs the measured target points (green) as the camera parameters are iteratively improved throughout the calibration process using a synthetic dataset.

The more closely the magenta and green points match, the more accurate the calibration parameters are.


## Introduction

Camera calibration is the process of **computing a finite set of camera parameters** which give a spatial understanding of a scene from 2D images.


Camera parameters can be divided into the following two groups:

- **Intrinsic parameters**: these map a 3D scene into the 2D coordinates of the camera sensor, the 'projection' of a 3D scene in to a 2D image.
- **Extrinsic parameters**: the 'pose' of the camera in the 'world' coordinate system (aka the transform from 'camera' coordinates to 'world' coordinates).

Generally, camera calibration is done through analyzing multiple images of a physical calibration target where either the camera or calibration target (or both) are in different positions.

You'll find all the code I used for this here: [github.com/pvphan/camera-calibration](https://github.com/pvphan/camera-calibration)


## Motivation

It's easy enough to call a library to do camera calibration without understanding fully what it does.
The steps are:
- Collect a set of images using the camera you want to calibrate.
- Extract the 2D calibration target points from each image (in image pixel coordinates).
- Specify the 3D arangement of these target points (in target 3D coordinates).
- Call your solver (e.g. cv2.calibrateCamera)

So why go through the effort of reimplementing this algorithm from scratch?

Personally, my reasons are:
1. To fill the cracks in my knowledge in what information about the scene is truely required for an accurate calibration.
2. As an exercise to brush up my linear algebra and optimization skills.

If that's good enough for you too, read on!


## High level algorithm
General flow


## The math
The minimal set of math 'bag of tricks' to know, and a simple description of what they do and why it's reasonable to expect they will work.


## The implementation
What my code does


### Using Levenberg-Marquardt
Using sympy to automatically differentiate


## Conclusion
Why did I do this
What I hope you got out of this
