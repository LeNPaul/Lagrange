---
layout: post
title: "Primer on camera calibration, part 2/2"
author: "Paul Vinh Phan"
categories: journal
image: thedraughtsmanofthelute.jpeg
tags: [camera,calibration,intrinsic,extrinsic,optimization,levenberg-marquardt]
---


# What is 'Zhang's method'?

Currently, the most popular method for calibrating a camera is **Zhang's method** published in [A Flexible New Technique for Camera Calibration](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf) by Zhengyou Zhang (1998).
Older methods typically required a precisely made 3D calibration target or a mechanical system to precisely move the camera or target.
In contrast, Zhang's method requires only a 2D calibration target and only loose requirements on how the camera or target moves.
This means that anyone with a desktop printer and a little time can accurately calibrate their camera!

The general strategy of Zhang's method is to impose naïve assumptions as constraints to get an **initial guess** of parameter values with singular value decomposition (SVD), then release those constraints and **refine** those guesses with non-linear least squares optimization.

The ordering of steps for Zhang's method are:
1. Use the 2D-3D point associations to compute an *initial guess* for the **intrinsic matrix**, $$A_{init}$$.
2. Using the above, compute an *initial guess* for the **distortion parameters**, $$\textbf{k}_{init}$$.
3. Using the above, compute an *initial guess* **camera pose** (per-view) in target coordinates, $$\textbf{W}_{init}$$.
4. Initialize **non-linear optimization** with the *initial guesses* above and then **iterate** to minimize **projection error**, producing $$A_{final}$$, $$\textbf{k}_{final}$$, and $$\textbf{W}_{final}$$.


# The steps of Zhang's method

Below, we'll be changing between vector/matrix formulations of equations and their scalar value forms.
Though less compact, it's crucial so that we can rewrite the equations into ones that can be solved with powerful techniques like SVD or non-linear least squares optimization.

## Zhang.1) Compute initial intrinsic matrix, A

First, we need to use the 2D-3D point associations to compute the homographies $$\textbf{H} = [H_1, H_2, ..., H_n]$$, for each of the $$n$$ views in the dataset.

To do this we employ two tricks:
- Trick #1: Assume there is **no distortion** in the camera (7.a). This is of course not true, but it will let us compute approximate values.
- Trick #2: We define the world coordinate system so that it's $$z = 0$$ plane is the **plane of the calibration target**.
This allows us to drop the $$z$$ term of the 3D world points (7.b).

Trick #1 simplifies our projection equation (5) to a distortion-less [**pinhole camera model**](https://hedivision.github.io/Pinhole.html):

$$
\begin{equation}
s \cdot u_{ij}
=
\textbf{A}
\cdot
\Pi \cdot W_i \cdot {}^wX_{ij}
\tag{7.a}\label{eq:7.a}
\end{equation}
$$

And after expanding, trick #2 allows us to drop some dimensions from this expression:

$$
\begin{equation}
s
\begin{pmatrix}
u\\
v\\
1\\
\end{pmatrix}
_{ij}
=
\textbf{A}
\cdot
\begin{pmatrix}
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 1 & 0\\
\end{pmatrix}
\begin{pmatrix}
|     & |     & |     & t_x\\
r_{x} & r_{y} & r_{z} & t_y\\
|     & |     & |     & t_z\\
0 & 0 & 0 & 1\\
\end{pmatrix}
_{i}
\begin{pmatrix}
x_w\\
y_w\\
0\\
1\\
\end{pmatrix}
_{ij}
\tag{7.b}\label{eq:7.b}
\end{equation}
$$

$$
s
\begin{equation}
\begin{pmatrix}
u\\
v\\
1\\
\end{pmatrix}
_{ij}
=
\underbrace{
    \textbf{A}
    \cdot
    \begin{pmatrix}
    |     & |     & t_x\\
    r_{x} & r_{y} & t_y\\
    |     & |     & t_z\\
    \end{pmatrix}
    _{i}
}_{H_i}
\begin{pmatrix}
x_w\\
y_w\\
1\\
\end{pmatrix}
_{ij}
\tag{8}\label{eq:8}
\end{equation}
$$

We've now arrived at an expression for our homography $$H_i$$ which relates a 2D point on the target plane to a 2D point on our image plane for the $$i$$-th view:

$$
\begin{equation}
s
\begin{pmatrix}
u\\
v\\
1\\
\end{pmatrix}
_{ij}
=
H_i
\begin{pmatrix}
x_w\\
y_w\\
1\\
\end{pmatrix}
_j
\tag{9}\label{eq:9}
\end{equation}
$$

3. Normalize the input datasets

```python
def estimateHomography(Xa: np.ndarray, Xb: np.ndarray):
    """
    Estimate homography using DLT
    Inputs:
        Xa -- 2D points in sensor
        Xb -- 2D model points
    Output:
        aHb -- homography matrix which relates Xa and Xb
    Rearrange into the formulation:
        M * h = 0
    M represents the model and sensor point correspondences
    h is a vector representation of the homography aHb we are trying to find:
        h = (h11, h12, h13, h21, h22, h23, h31, h32, h33).T
    """
    N = Xa.shape[0]
    M = np.zeros((2*N, 9))
    for i in range(N):
        ui, vi = Xa[i][:2]
        Xi, Yi = Xb[i][:2]
        M[2*i,:]   = (-Xi, -Yi, -1,   0,   0,  0, ui * Xi, ui * Yi, ui)
        M[2*i+1,:] = (  0,   0,  0, -Xi, -Yi, -1, vi * Xi, vi * Yi, vi)
    U, S, V_T = np.linalg.svd(M)
    h = V_T[-1]
    Hp = h.reshape(3,3)
    aHb = Hp / Hp[2,2]
    return aHb
```

## Zhang.2) Compute initial distortion vector, k


## Zhang.3) Compute initial extrinsic parameters, W


## Zhang.4) Refine A, k, W using non-linear optimization

Until now, we've been estimating $$\textbf{A}, \textbf{k}, \textbf{W}$$ to get a good initialization point for our optimization.
In non-linear optimization, it's often impossible to arrive at a good solution unless the initialization point for the variables was at least somewhat reasonable.

1. Express the projection equation symbolically (e.g. `sympy`).
1. Take the partial derivatives of the projection expression with respect to the calibration parameters.
1. Arrange these partial derivative expressions into the Jacobian matrix $$J$$ for the projection expression.

Now we are ready to run our non-linear optimization algorithm. Levenberg-Marquardt is a popular choice as it works well in practice.

1. Start by setting the *current* calibration parameters $$\textbf{P}_{curr}$$ to the initial guess values computed in Zhang.1 - Zhang.3.
1. Use $$\textbf{P}_{curr}$$ to project the input world points $${}^wX_{ij}$$ to thier image coordinates $$u_{ij}$$
1. Evaluate the Jacbobian $$J$$ for all input points at the *current* calibration parameter values.

Below, green crosses are the measured 2D marker points and magenta crosses are the projection of the associated 3D points using the 'current' camera parameters.
This gif plays through the iterative refinement of the camera parameters (step #5 of Zhang's method).
(Generation of this gif is part of the github repo linked at the top of this post.)

![](assets/img/reprojection.gif)
{: centeralign }


# Final remarks

In my experience, camera calibration can be daunting due to assumed knowledge in several areas (camera projection models, linear algebra, optimization) and how those areas intersect.
We walked through a common camera projection model and then stepped through Zhang's method, introducing numerical methods as needed.

I hope this post has demystified `cv2.calibrateCamera()` a little bit.
Thanks for reading!


# Appendix


### Rigid-body transformations

Recall we defined $$\textbf{W} = [W_1, W_2, ..., W_n]$$, where $$W_i$$ is the $$i$$-th **rigid-body transform** from *world* to *camera*, which is also the **pose** of the *world* in *camera* coordinates.

This can also be written in what I've been told is the 'Craig convention': $$\textbf{W} = [{}^cM_{w,1}, {}^cM_{w,2}, ..., {}^cM_{w,N}]$$, where $${}^cM_{w,i}$$ is the $$i$$-th **rigid-body transform** from  *world* to *camera*, which is also the **pose** of the *world* in *camera* coordinates

Each transform expressed in homogeneous form:

$$
{}^cM_{w}
=
\begin{pmatrix}
|     & |     & |     & t_x\\
r_{x} & r_{y} & r_{z} & t_y\\
|     & |     & |     & t_z\\
0 & 0 & 0 & 1\\
\end{pmatrix}
$$

- $$t_x, t_y, t_z$$ are the world coordinate system's **origin** given *in the camera's coordinates*
- $$r_x$$ (3x1 column vector) is the **normalized direction vector** of the world coordinate system's x-axis given *in the camera's coordinates* ($$r_y$$, $$r_z$$ follow this pattern)

Notational example of transforming a single, homogeneous 3D point: \
$${}^cP = {}^cM_{w} \cdot {}^wP$$, with $$P \in \mathbb{R^3}$$ and $${}^cM_{w} \in SE(3)$$

- $${}^cP$$ --- homogeneous point $$P$$ in camera coordinates,
$$
\begin{pmatrix}
x_c & y_c & z_c & 1
\end{pmatrix}
^\top$$

- $${}^wP$$ --- homogeneous point $$P$$ in world coordinates,
$$
\begin{pmatrix}
x_w & y_w & z_w & 1
\end{pmatrix}
^\top$$

- $$\mathbb{R^3}$$ --- the space of real, 3 dimensional numbers
- $$SE(3)$$ --- $$S$$pecial $$E$$uclidean group 3, the space of 3D rigid-body transformations


### Singular Value Decomposition (SVD)

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


### Non-linear least squares optimization (Levenberg-Marquardt)

Non-linear optimization is the task of computing a set of parameters which **minimizes a non-linear value function**.
This is a huge topic of its own that I'm not quite ready to articulate.
Maybe I'll do a post in the future and update this one with a link.

![](https://i.stack.imgur.com/gdJ3v.gif)
{: centeralign }

Visualization of Gauss-Newton optimization.
{: centeralign }

Additional links:
- [Algorithms for Optimization (Kochenderfer & Wheeler)](https://mitpress.mit.edu/books/algorithms-optimization) (yay, Tim!)


Substituting the definition of predicted position $$u_{ij}$$ from (5):

$$
\begin{equation}
E
=
\sum\limits_{i}^{n} \sum\limits_{j}^{m}

||
z_{ij}
-
hom^{-1}
(
    \textbf{A}
    \cdot
    hom(
        distort(
            hom^{-1}(
                \Pi \cdot W_i \cdot {}^wX_{ij}
            ),
            \textbf{k}
        )
    )
)
||^2
\tag{6.b}\label{eq:6.b}
\end{equation}
$$

