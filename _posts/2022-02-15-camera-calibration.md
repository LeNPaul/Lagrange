---
layout: post
title: "Camera calibration from scratch"
author: "Paul Vinh Phan"
categories: journal
image: reprojection.gif
tags: [camera,calibration,intrinsic,extrinsic,optimization,levenberg-marquardt]
---


## What is camera calibration?

A camera captures light from a 3D scene and projects it onto a 2D sensor which stores the sensor state as a 2D image.
A **2D point** in the image is thus equivalent to a **3D ray** in the scene.
A camera is **calibrated** if we know the *camera parameters* which define the mapping between these spaces.
**Camera calibration** is the process of computing the camera parameters:
- $A$ -- the intrinsic matrix,
$$
\begin{pmatrix}
\alpha & \gamma & c_x\\
0 & \beta & c_y\\
0 & 0 & 1
\end{pmatrix}
$$
    - $\alpha$ -- focal length x
    - $\beta$ -- focal length y
    - $\gamma$ -- the skew ratio
    - $c_x$ -- x coordinate of principal point (optical center)
    - $c_y$ -- y coordinate of principal point (optical center)
- $\textbf{k}$ -- the distortion vector,
    - for the radial-tangential model:
$$
\begin{pmatrix}
k_1 & k_2 & p_1 & p_2 & k_3
\end{pmatrix}
$$
    - for the fisheye model:
$$
\begin{pmatrix}
k_1 & k_2 & k_3 & k_4
\end{pmatrix}
$$
    - $k_i$ values correspond to radial distortion and $p_i$ values correspond to tangential distortion
- $\textbf{W}$ -- the per-view set of transforms from target to camera (list of N 4x4 matrices)
    - $\textbf{W} = [W_1, W_2, ..., W_N]$, where $W_i$ is the transform from 'world' (the calibration target) to 'camera' for the $i$-th view.
    - Written in homogenous form: $W_i = $
$$
\begin{pmatrix}
R_{11} & R_{12} & R_{13} & t_1\\
R_{21} & R_{22} & R_{23} & t_2\\
R_{31} & R_{32} & R_{33} & t_3\\
0 & 0 & 0 & 1\\
\end{pmatrix}
$$
    - Note on convention: $W_i = {}^cM_{w,i}$ which is the $i$-th **transform** from *world* to *camera*, which is also the **pose** of the *world* in *camera* coordinates. Here, $M$ simply denotes a 4x4 homogeneous rigid body transformation.

A camera calibration **dataset** is gathered by capturing multiple images of a known physical calibration target, changing the camera pose or board pose for each view.
While calibrating the *intrinsic parameters*, we'll also consider the *extrinsic parameters* which are simply the rigid body transforms from the calibration target to the camera coordinate frame for each view.

The problem of camera calibration is: "Given N images of a known calibration target, compute $A$, $\textbf{k}$, and $\textbf{W}$."


## Why from scratch?

I had used [cv2.calibrateCamera()](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d) many times without understanding why it sometimes failed or gave poor results.
It was also a good exercise to deepen my linear algebra and optimization understanding.

My 'from scratch' implementation of Zhang's method is linked here: [github.com/pvphan/camera-calibration](https://github.com/pvphan/camera-calibration)


## What is 'Zhang's method'?

Currently, the most popular method for calibrating a camera is **Zhang's method** invented by [Zhengyou Zhang (1998), A Flexible New Technique for Camera Calibration](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)).
Older methods typically required a precisely made 3D calibration target or a mechanical system to precisely move the camera.
In contrast, Zhang's method requires only a 2D calibration target and only loose requirements on how the camera or target moves.
This meant that anyone with a desktop printer and a little time could now accurately calibrate their camera!

We will assume the 2D target points have already been extracted and have known association with the 3D target points (in the target's coordinate system). The following steps for Zhang's method are:
1. Use the 2D-3D point associations to **compute the homography** (per-view) from target to camera.
2. Use the homographies to compute an initial guess for the **intrinsic parameter** matrix.
3. Using the above, compute an initial guess for the **distortion parameters**.
4. Using the above, compute an estimated **camera pose** (per-view) in target coordinates.
5. Using the above, **refine** the camera poses, intrinsic parameters, and distortion parameters using **nonlinear optimization** to minimize reprojection error.


## The math
The minimal set of math 'bag of tricks' to know, and a simple description of what they do and why it's reasonable to expect they will work.


### Singular Value Decomposition (SVD)


### Non-linear optimization: Levenberg-Marquardt

Non-linear optimization is the task of computing a set of parameters which minimizes a non-linear value function.
In contrast, a linear optimization method (e.g. least-squares) could solve for $m$ and $b$ for a linear function $y = m*x + b$, but not for a non-linear function such as $y = m*x^2 + sin(b)$

Levenberg-Marquardt (LM) is a technique for nonlinear optimization.
LM is based on the Gauss-Newton method but with a tweak to it's weighting which works well in practice (read more on this more here).


## The implementation
What my code does

The gif above shows the reprojection of expected target points (magenta) vs the measured target points (green) as the camera parameters are iteratively improved throughout the calibration process using a synthetic dataset.

The more closely the magenta and green points match, the more accurate the calibration parameters are.


## Conclusion

Here, I implemented Zhang's camera calibration method mostly from scratch.
The full pipeline of these techniques was a bit daunting to me taken altogether, but after implementing it piece by piece I found the Zhang method to be elegant in it's clever use of SVD and Levenberg-Marquardt.

I hope this post has helped some people become more comfortable with these linear algebra and optimization techniques and demystified Zhang's method a little bit.
Thanks for reading!
