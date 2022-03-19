---
layout: post
title: "Camera calibration overview"
author: "Paul Vinh Phan"
categories: journal
tags: [camera,calibration,intrinsic,extrinsic,optimization,levenberg-marquardt]
---

{:centeralign: style="text-align: center;"}

Below is an overview of the theory behind Zhang's camera calibration method.
For those interested in implementation, here's my Python version: [github.com/pvphan/camera-calibration](https://github.com/pvphan/camera-calibration).


## What is camera calibration?

A camera captures light from a 3D scene and projects it onto a 2D sensor which stores the sensor state as a 2D image.
In other words, a **2D point** in the image is equivalent to a **3D ray** in the scene.
A camera is **calibrated** if we know the *camera parameters* which define the mapping between these spaces.
The 'pinhole camera model' is and idealized way to illustrate this, though it does not factor in lens distortion.

![](https://docs.opencv.org/4.x/pinhole_camera_model.png)
{: centeralign }
Illustration of the 'pinhole camera model' ([OpenCV 'calib3d' documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#details))
{: centeralign }

Camera calibration is the process of computing the **camera parameters**: $$A$$, $$\textbf{k}$$, and $$\textbf{W}$$ which are further discussed in the [$$\S$$Camera parameters](#camera-parameters) section.
A camera calibration **dataset** is gathered by capturing multiple images of a known physical calibration target and varying the board pose with respect to the camera for each view.

So, given multiple images of a known calibration target, camera calibration computes the camera parameters: $$A$$, $$\textbf{k}$$, and $$\textbf{W}$$.
And with these parameters, we can **reason spatially** about the world from images!

![](assets/img/pict_calib_mini2.gif)
{: centeralign }

A calibration dataset and its visualization ([vision.caltech.edu](http://www.vision.caltech.edu/bouguetj/calib_doc/))
{: centeralign }


## Camera parameters

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
    - $$\textbf{W} = [W_1, W_2, ..., W_n]$$, where $$W_i$$ is the $$i$$-th **rigid-body transform** *world* to *camera*, which is also the **pose** of the *world* in *camera* coordinates. (See the [$$\S$$Appendix](#appendix) for more discussion on convention).


## Aside: detecting target points in 2D images

For the remainder of this post, we will assume the 2D target points (i.e. pixel coordinates) have already been detected in the images and have known association with the 3D target points in the target's coordinate system.
Such functionality is typically handled by a library (e.g. [ChArUco](https://docs.opencv.org/3.4/df/d4a/tutorial_charuco_detection.html), [AprilTag](https://april.eecs.umich.edu/software/apriltag)) and is beyond the scope of this post.


![](https://docs.opencv.org/3.4/charucodefinition.png)
{: centeralign }

The corners of the larger checkerboard are the points which are detected ([OpenCV.org](https://docs.opencv.org/3.4/df/d4a/tutorial_charuco_detection.html))
{: centeralign }


## What is 'Zhang's method'?

Currently, the most popular method for calibrating a camera is **Zhang's method** published in [A Flexible New Technique for Camera Calibration](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf) by Zhengyou Zhang (1998).
Older methods typically required a precisely made 3D calibration target or a mechanical system to precisely move the camera.
In contrast, Zhang's method requires only a 2D calibration target and only loose requirements on how the camera or target moves.
This means that anyone with a desktop printer and a little time can accurately calibrate their camera!

The general strategy of Zhang's method is to impose naïve assumptions as constraints to get an **initial guess** of parameter values with singular value decomposition (SVD), then release those constraints and **refine** those guesses with non-linear least squares optimization.

The ordering of steps for Zhang's method are:
1. Use the 2D-3D point associations to **compute the homography** (per-view) from target to camera.
2. Use the homographies to compute an *initial guess* for the **intrinsic matrix**, $$A_{init}$$.
3. Using the above, compute an *initial guess* for the **distortion parameters**, $$\textbf{k}_{init}$$.
4. Using the above, compute an *initial guess* **camera pose** (per-view) in target coordinates, $$\textbf{W}_{init}$$.
5. Initialize **nonlinear optimization** with the *initial guesses* above to minimize **projection error**, producing $$A_{final}$$, $$\textbf{k}_{final}$$, and $$\textbf{W}_{final}$$.

![](https://media1.giphy.com/media/NsIwMll0rhfgpdQlzn/giphy.gif)
{: centeralign }


## Projection error: the metric of calibration 'goodness'

In order to compute camera parameters which are useful for spatial reasoning, we need to define what makes a set of parameters better than another set.
This is typically done by computing **sum-squared projection error**, $$E$$.
The lower that error metric is, the more closely our camera parameters fit the measurements from the input images.
- From each image, we have the detected marker points. Each marker point is a single **2D measurement**, which we will denote as $$z_{ij}$$ for the $$j$$-th measured point of the $$i$$-th image.
- From each measurement $$z_{ij}$$, we also have the **corresponding 3D point** in target coordinates (known by construction), which we will denote as $$X_{ij}$$.
- With a set of calibration parameters ($$A$$, $$\textbf{k}$$, $${}^cM_{w,i}$$), we can then project where that 3D point should appear in the 2D image -- a single **2D prediction**, which we will express as the distorted-projected point, $$\tilde{x}_{ij}$$.
- The Euclidean distance between the 2D prediction and 2D measurement is the **projection error** for a single point.

Considering the full dataset, we can compute the sum-squared projection error by computing the Euclidean distance between each (*measurement*, *prediction*) pair for all $n$ images and all $m$ points in those images:

$$
E = \sum\limits_{i}^{n} \sum\limits_{j}^{m} || z_{ij} - \tilde{x}_{ij} ||^2
$$

![](assets/img/projectionerror.png)
{: centeralign }

Illustration of projection error for a single measurement ($$z_{ij}$$) and prediction ($$\tilde{x}_{ij}$$) pair.
{: centeralign }

The projection function can be expressed as:

$$
\tilde{x}_{ij} = A \cdot distort(x_{ij}, \textbf{k})
$$

$$
x_{ij} = \Pi \cdot {}^cM_{w,i} \cdot X_{ij}
$$

Below, green crosses are the measured 2D marker points and magenta crosses are the projection of the associated 3D points using the 'current' camera parameters.
This gif plays through the iterative refinement of the camera parameters (step #5 of Zhang's method).

![](assets/img/reprojection.gif)
{: centeralign }

## Numerical toolbelt

We'll need some numerical methods in our toolbelt, two of the major ones are overviewed below.
I'll not go into great detail about these methods, but I'll leave links to explore them further.

### 1. Singular Value Decomposition (SVD)

SVD decomposes a matrix $M$ ($m$,$n$) to three matrices $U$ ($m$,$m$), $\Sigma$ ($m$,$n$), and $V^\top$ ($n$,$n$), such that $$M = U \cdot \Sigma \cdot V^\top$$.
The properties of the resulting three matrices are like that of Eigenvalue and Eigenvector decomposition.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Singular_value_decomposition_visualisation.svg/206px-Singular_value_decomposition_visualisation.svg.png)
{: centeralign }

Visualization of SVD from Wikipedia.
{: centeralign }

The properties of this decomposition have many uses, one of which is **solving homogeneous linear systems** of the form
$$M \cdot x = 0$$.

A solution for the value of $x$ for this linear system is the smallest eigenvector of $V^\top$.
Using numpy, solving looks like this:

```python
# M * x = 0, where M (m,n) is known and we want to solve for x (n,1)
U, Σ, V_T = np.linalg.svd(M)
x = V_T[-1]
```

Additional links:
- [(Wikipedia) More applications of the SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition#Applications_of_the_SVD)
- [(blog) Explanation of SVD by Greg Gunderson](https://gregorygundersen.com/blog/2018/12/10/svd/)


### 2. Non-linear least squares optimization (Levenberg-Marquardt)

Non-linear optimization is the task of computing a set of parameters which **minimizes a non-linear value function**.
TODO

![](https://i.stack.imgur.com/gdJ3v.gif)
{: centeralign }

Visualization of Gauss-Newton optimization.
{: centeralign }


## The implementation, step by step


## Final remarks

We discussed an implemention of Zhang's camera calibration method mostly from scratch.
Though initially daunting, I found implementing each step piece by piece made the whole process more digestible.

I hope this post has helped some other people become more comfortable with SVD and LM, and demystified `cv2.calibrateCamera()` a little bit.
Thanks for reading!


## Appendix

- Recall we defined $$\textbf{W} = [W_1, W_2, ..., W_n]$$, where $$W_i$$ is the $$i$$-th **rigid-body transform** *world* to *camera*, which is also the **pose** of the *world* in *camera* coordinates.
    - This can also be written in what I've been told is the 'Craig convention': $$\textbf{W} = [{}^cM_{w,1}, {}^cM_{w,2}, ..., {}^cM_{w,N}]$$, where $${}^cM_{w,i}$$ is the $$i$$-th **rigid-body transform** *world* to *camera*, which is also the **pose** of the *world* in *camera* coordinates

    - Each transform expressed in homogeneous form: $${}^cM_{w} = $$
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
    - Notational example of transforming a single, homogeneous 3D point: \
$${}^cP = {}^cM_{w} \cdot {}^wP$$, with $$P \in \mathbb{R^3}$$ and $${}^cM_{w} \in SE(3)$$
        - $${}^cP$$ -- homogeneous point $$P$$ in camera coordinates,
$$
\begin{pmatrix}
x_c & y_c & z_c & 1
\end{pmatrix}
^\top$$
        - $${}^wP$$ -- homogeneous point $$P$$ in world coordinates,
$$
\begin{pmatrix}
x_w & y_w & z_w & 1
\end{pmatrix}
^\top$$
        - $$\mathbb{R^3}$$ -- the space of real, 3 dimensional numbers
        - $$SE(3)$$ -- $$S$$pecial $$E$$uclidean group 3, the space of 3D rigid-body transformations
