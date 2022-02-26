---
layout: post
title: "Camera calibration from scratch"
author: "Paul Vinh Phan"
categories: journal
image: reprojection.gif
tags: [camera,calibration,intrinsic,extrinsic,optimization,levenberg-marquardt]
---

Above, green crosses are the measured 2D marker points and magenta crosses are the projection of the associated 3D points using the 'current' calibration parameters (iterative).
My 'from scratch' implementation of Zhang's camera calibration: [github.com/pvphan/camera-calibration](https://github.com/pvphan/camera-calibration).

## What is camera calibration?

A camera captures light from a 3D scene and projects it onto a 2D sensor which stores the sensor state as a 2D image.
In other words, a **2D point** in the image is equivalent to a **3D ray** in the scene.
A camera is **calibrated** if we know the *camera parameters* which define the mapping between the 2D image point / 3D ray spaces.
Camera calibration is the process of computing the **camera parameters**: $$A$$, $$\textbf{k}$$, and $$\textbf{W}$$ (defined in [$$\S$$Camera parameters](#camera-parameters)).

A camera calibration **dataset** is gathered by capturing multiple images of a known physical calibration target, varying the camera pose and/or board pose for each view for a total of N views.

So, **given N images of a known calibration target, compute the camera parameters: $$A$$, $$\textbf{k}$$, and $$\textbf{W}$$.** And with these parameters, we can **reason spatially** about the world from images!

{:centeralign: style="text-align: center;"}
![Figure 1](assets/img/pict_calib_mini2.gif)
{: centeralign }

{:centeralign: style="text-align: center;"}
A calibration dataset and its visualization, from [vision.caltech.edu](http://www.vision.caltech.edu/bouguetj/calib_doc/).
{: centeralign }


## Why from scratch?

I have used [cv2.calibrateCamera()](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d) many times without understanding why it sometimes fails or gives poor results.
Having a better understanding of the fundamentals will make it more obvious to me if something about the setup is ill-posed.
Writing the code myself was also a good exercise to deepen my linear algebra and optimization understanding.


## What is 'Zhang's method'?

Currently, the most popular method for calibrating a camera is **Zhang's method** invented in [A Flexible New Technique for Camera Calibration](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf) by Zhengyou Zhang (1998).
Older methods typically required a precisely made 3D calibration target or a mechanical system to precisely move the camera.
In contrast, Zhang's method requires only a 2D calibration target and only loose requirements on how the camera or target moves.
This means that anyone with a desktop printer and a little time can accurately calibrate their camera!

For the remainder of this post, we will assume the 2D target points have already been extracted from the images and have known association with the 3D target points (in the target's coordinate system).
Such functionality is typically handled by a library (e.g. [ChAruco](https://docs.opencv.org/3.4/df/d4a/tutorial_charuco_detection.html), [AprilTag](https://april.eecs.umich.edu/software/apriltag)) and is beyond the scope of this post.

The general strategy of Zhang's method is to impose naïve assumptions as constraints to get an **initial guess** of parameter values with singular value decomposition (SVD), then release those constraints and **refine** those guesses with non-linear least squares optimization.

The ordering of steps for Zhang's method are:
1. Use the 2D-3D point associations to **compute the homography** (per-view) from target to camera.
2. Use the homographies to compute an initial *guess* for the **intrinsic matrix**.
3. Using the above, compute an initial *guess* for the **distortion parameters**.
4. Using the above, compute an initial *guess* **camera pose** (per-view) in target coordinates.
5. Using the above, **refine** the guess camera poses, intrinsic parameters, and distortion parameters using **nonlinear optimization** to minimize projection error.


## Camera parameters

A more full definition of the calibration parameters is provided below. The variable naming convention follows fairly closely to the Zhang paper.
- $$A$$ -- the **intrinsic matrix**,
$$
\begin{pmatrix}
\alpha & \gamma & u_0\\
0 & \beta & v_0\\
0 & 0 & 1\\
\end{pmatrix}
$$
    - $$\alpha$$ -- focal length in the camera x direction
    - $$\beta$$ -- focal length in the camera y direction
    - $$\gamma$$ -- the skew ratio, typically 0
    - $$u_0$$ -- u coordinate of optical center in image coordinates
    - $$v_0$$ -- v coordinate of optical center in image coordinates
- $$\textbf{k}$$ -- the **distortion vector**,
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
    - $$k_i$$ values correspond to radial distortion and $$p_i$$ values correspond to tangential distortion
- $$\textbf{W}$$ -- the **per-view set of transforms** (also called **extrinsic** parameters) from target to camera, which is a list of N 4x4 matrices
    - $$\textbf{W} = [{}^cM_{w,1}, {}^cM_{w,2}, ..., {}^cM_{w,N}]$$, where $${}^cM_{w,i}$$ is the $$i$$-th 4x4 homogeneous rigid-body **transform** (i.e. in the group $$SE(3)$$) from *world* to *camera*, which is also the **pose** of the *world* in *camera* coordinates
    - Written in homogeneous form: $${}^cM_{w} = $$
$$
\begin{pmatrix}
|     & |     & |     & t_x\\
r_{x} & r_{y} & r_{z} & t_y\\
|     & |     & |     & t_z\\
0 & 0 & 0 & 1\\
\end{pmatrix}
$$
        - $$t_x, t_y, t_z$$ are the world coordinate system's origin given in the camera coordinates
        - $$r_x$$ (3x1 column vector) is the normalized direction vector of the world coordinate system's x-axis given in camera coordinates ($$r_y$$, $$r_z$$ follow this pattern)
    - Example of transforming a single, homogeneous 3D point: $${}^cP = {}^cM_{w} \cdot {}^wP$$, with $$P \in \mathbb{R^3}$$ and $${}^cM_{w} \in SE(3)$$
        - $${}^wP$$ -- homogeneous point $$P$$ in world coordinates, e.g.
$$
\begin{pmatrix}
x & y & z & 1
\end{pmatrix}
^\top$$
        - $${}^cP$$ -- homogeneous point $P$ in camera coordinates
        - $$\mathbb{R^3}$$ -- the space of real, 3 dimensional numbers

The minimal set of math 'bag of tricks' to know, and a simple description of what they do and why it's reasonable to expect they will work.


## Numerical toolbelt

So we'll need those two numerical methods in our toolbelt:

### 1. Singular Value Decomposition (SVD)

TODO


### 2. Non-linear optimization using the Levenberg-Marquardt (LM) algorithm

TODO
Non-linear optimization is the task of computing a set of parameters which minimizes a non-linear value function.
LM is a technique for nonlinear optimization based on the Gauss-Newton method.
It's tweak to it's weighting which works well in practice (more on this here).


## The implementation
TODO: What my code does

The gif above shows the projection of expected target points (magenta) vs the measured target points (green) as the camera parameters are iteratively improved throughout the calibration process using a synthetic dataset.

The more closely the magenta and green points match, the more accurate the calibration parameters are.


## Final remarks

We discussed an implemention of Zhang's camera calibration method mostly from scratch.
Though initially daunting, I found implementing each step piece by piece made the whole process more digestible.

I hope this post has helped some other people become more comfortable with SVD and LM, and demystified Zhang's method a little bit.
Thanks for reading!

