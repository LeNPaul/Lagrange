---
layout: post
title: "Camera Calibration using Zhang's Method"
author: "Paul Vinh Phan"
categories: journal
image: reprojection.gif
tags: [camera,calibration,intrinsic,extrinsic,optimization,levenberg-marquardt]
---

The gif above shows the reprojection of expected target points (magenta) vs the measured target points (green) as the camera parameters are iteratively improved throughout the calibration process using a synthetic dataset.

The more closely the magenta and green points match, the more accurate the calibration parameters are.


## Introduction

Camera calibration is the process of **computing a finite set of camera parameters** which give a spatial understanding of a scene from 2D images.


Camera parameters can be divided into the following two groups:

- **Intrinsic parameters**: these map a 3D scene into the 2D coordinates of the camera sensor, the 'projection' of a 3D scene in to a 2D image.
- **Extrinsic parameters**: the 'pose' (position and orientation) of the camera in the world coordinate system (aka the transform from 'camera' coordinates to 'world' coordinates).

Generally, camera calibration is done through analyzing multiple images of a physical calibration target where either the camera or calibration target (or both) are in different positions.

You'll find all the code I used for this here: [github.com/pvphan/camera-calibration](https://github.com/pvphan/camera-calibration)


## Motivation

It's easy enough to call a library to do camera calibration without understanding fully what it does.
The steps are:
1. Collect a set of images using the camera you want to calibrate.
2. Extract the 2D calibration target points from each image (in image pixel coordinates).
3. Specify the 3D arangement of these target points (in target 3D coordinates).
4. Call your solver (e.g. cv2.calibrateCamera)

So why go through the effort of reimplementing this algorithm from scratch?

Personally, my reasons are:
- To fill the cracks in my knowledge in what information about the scene is truely required for an accurate calibration.
- As an exercise to deepen my linear algebra and optimization understanding.

As such, I've decided to focus just on step 4 above. If that's good enough for you too, read on!


## Overview

Since about 1990, the most popular method for calibrating a camera is the Zhang method invented by Zhenyou Zhang.
It's popularity is due to how easy and inexpensive it is compared to older methods.
Older methods typically have stricter requirements such as a precisely made 3D calibration target or a sysem to precisely move the camera.
In contrast, Zhang's method requires only a 2D calibration target and only loose requirements on how the camera or target moves.
This target could be printed on any normal desktop printer and taped to something reasonably flat such as a clipboard or cardboard box and still provide very accurate results.

We will assume the 2D target points have already been extracted and have known association with the 3D target points (in the target's coordinate system).


## The math
The minimal set of math 'bag of tricks' to know, and a simple description of what they do and why it's reasonable to expect they will work.


### Singular Value Decomposition (SVD)


### Levenberg-Marquard


## The implementation
What my code does


## Conclusion

Here, I implemented Zhang's camera calibration method mostly from scratch.
This grounds general camera calibration process into the clever application of a few mathematical techniques (SVD, Levenberg-Marquardt).
The full pipeline of these techniques was a bit daunting taken altogether, but after implementing it piece by piece I found the Zhang method to be elegant in it's clever use of just a couple powerful techniques.

I hope this post has helped some people become more comfortable with these linear algebra and optimization techniques and demystified Zhang's method a little bit.
Thanks for reading!
