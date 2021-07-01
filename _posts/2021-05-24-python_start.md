---
layout: post
title: "Reconstructing Climate data Using Python"
author: "Dani Lafarga"
categories: journal
tags: [documentation,sample]
---


## Motivation
Sea surface temperature (SST) is studied more often than deep sea temperatures. More recently there has been more interest in deep ocean temperatures, but there has been a limitation to research due to the lack of measurement in those areas. The objective of this NERTO research is to make a high-resolution reconstruction of deep ocean temperatures of depths up to 5,500 meters at ¼ degree spatial resolution and 5-day time resolution with 33 layers for 26 years.

## Background
### What is covariance?
Covariance measures how much two variables change. Typically in climatology covariance is considered between stations, grid boxes, or grid points. Covariance between two stations i and j can be denoted as:

$$ \sum _{ij}$$

For N stations covariance would be an N by N matrix:

$$ [\sum _{ij}]_{N\times N}$$

Consider data X that consists of N stations and Y time steps:
$$ X = [x_{it}]_{N\times Y}$$

Covarance can then be computed by first computing anomalies:
$$ A_{N \times Y} = [a_{it}]_{N\times Y} =  [x_{it} - \bar x_{i}]_{N\times Y}$$

Where:
$$\bar x_i = \frac1 Y \sum_{t=1} ^Y x_{it}$$
and recall:
$$ i = 1,2...N$$

This is the mean of the data or more formally climatology.

Covariance is computed by:
$$[\sum_{ij}]_{N\times N} = \frac1 Y A_{N\times Y} A_{Y\times N}^T$$

for this case becuase the data set is so large the  covariance is computed in terms of time rather than in terms of space. Instead of computing the covariance for each station we are computing  the covariance for every year given a specific month. 

$$[\sum_{tt'}]_{Y\times Y} = \frac1 N \sum_{i=1}^N a(t)a(t') = \frac1 N A_{Y\times N}^T A_{N\times Y}$$

from this we can compute emperical orthogonal functions.

### What are Empirical Orthogonal functions?
Emperical orthogonal function come from eigenvectors. Eigenvectors are vectors that point in the same direction as their corresponding matrix.

Consider the square covariance matrix $$[\sum_{tt'}]_{Y\times Y} $$ and some vector  w which runs parallel to $$[\sum_{tt'}]_{Y\times Y} $$ . There is a scalar or eigenvalue $$\lambda$$ which scales w such that:

$$[\sum _{tt'}]_{Y\times Y}  \vec{v} = \lambda \vec{v}$$
 
The first few eigenvectors of a large climate covariance matrix of climate data often represent some typical patterns of climate variability (Shen and Somerville 97). Usually EOFs are computed using singular value decomposition (SVD), but the method  used here is first finding covariance in time and then computing eigenvectors from that covariance matrix followed by multiplying the vectors to the anomaly matrix.

### Covariance in Time and EOFs 
Consider some data $$x_{it}$$ whose anomalies are $$ A_{N\times Y}$$ their spatial covariance is then:

$$ [C_{ij}]_{N\times N} = A_{N\times Y} A_{Y\times N}^T$$

and their covarience with respect to time is:

$$ [\sum_{tt'}]_{Y\times Y} = A_{Y\times N}^T A_{N\times Y}$$

In space there is some vector $$\vec{v}$$ that points in the same direction as $$ [C_{ij}]_{N\times N}$$ such that

$$ [C_{ij}]_{N\times N} \vec{u} = \rho \vec{u}$$

In time there is some other vector $$\vec{v}$$ that points in the same direction as $$[\sum _{tt'}]_{Y\times Y}$$ such that:

$$[\sum _{tt'}]_{Y\times Y}  \vec{v} = \lambda \vec{v}$$

Meaning $$\vec{u}$$ are the eigenvectors of  $$[C_{ij}]_{N\times N}$$ with $$\rho$$ being it's eigenvalues, and $$\vec{v}$$ are the eigenvectors of  $$[\sum _{tt'}]_{Y\times Y}$$ with $$\lambda$$ being it's eigenvalues. 

The problem we are trying to answer is how these things eigenvectors and eigenvalues relate? Above we defined covariance in time as
$$[\sum_{tt'}]_{Y\times Y} = \frac1 N A_{Y\times N}^T A_{N\times Y} $$. We can ignore the  1/N as this will only scale the unique vector and not change its direction. Therefore:

$$[\sum_{tt'}]_{Y\times Y} = A_{Y\times N}^T A_{N\times Y} $$

plugging this into its eigen value problem:

$$A_{Y\times N}^T A_{N\times Y} \vec{v} = \lambda \vec{v}$$

Multiply both sides by A:

$$A_{N\times Y} A_{Y\times N}^T A_{N\times Y} \vec{v} = \lambda A_{N\times Y}\vec{v}$$

because $$[C_{ij}]_{N\times N} = A_{N\times Y} A_{Y\times N}^T$$ we can redefine the equation above as:

$$[C_{ij}]_{N\times N} A_{N\times Y} \vec{v} = \lambda A_{N\times Y}\vec{v}$$

let $$\vec{w} = A_{N\times Y} \vec{v}$$ this gives:

$$[C_{ij}]_{N\times N}\vec{w} = \lambda\vec{w}$$

This is similar to:
$$ [C_{ij}]_{N\times N} \vec{u} = \rho \vec{u}$$

Comparing the two equations we can conclude: $$ \rho = \lambda$$ and $$ \vec{w} = A \vec{v}$$. $$ \vec{w}$$ does not equal to one so  it needs to be normalized. To do this  we divide it by its magnitude 

$$ EOF = \frac{A \vec{v}} {norm(A \vec{v})} $$

## The Data
The data used in the reconstruction of ocean temperatures is from JPL’s non-Boussinesq ocean general circulation model (OGCM). This data was initially collected on a $$1/4^\circ$$ by $$1/4^\circ$$ grid with ocean temperatures (in $$c^\circ$$) of multiple depths (in m) taken for 26 years. The data  is taken every 5 days making this quite a large problem for computing EOFs. The total amount of bytes needed to be read is:

$$1/4^\circ \times 1/4^\circ \times 33 \ layers \  \times 5 \ days$$

$$ = 1442 \times 698 \times 33 = 33,215,028 \ entries$$

there are 8 bytes for each dataum therefore there is:

$$ 8 \times 33,215,028 = 265MB \ per\ entry$$

The actual amount of bytes per entry is 274MB. There are 73 files for each year as each file represents a 5 day block and this is done for 26 years so there are 1901 files:

$$ \frac{365\ days}{year} \times \frac{1\ file}{5\ days} = \frac {73\ files}{years} \times 26 \ years = 1901 \ files$$
The total amount of bytes is:

$$ 1901 \ files \times 274MB = 521GB \times \ temperature \ and \ salinity = 521 \times 2 = 1.04TB \times 3 = 3TB$$

To model this data we start with a larger resolution and therefore a smaller amount of data. Consider then computing EOFs for each month from 1955-2003 for 33 depths of the ocean. On a one degree by one degree grid this would mean:

$$ 1 ^\circ \times 1^\circ \times 33 \ layers\ = 360 \times 180 \times 33 = 2,138,400\ Bytes\ of\ data\ per\ month$$

If there are 54 years needed to be modeled and there are 4 bytes per datum then:

$$
2MB \times 4B \times 54 \ years \times 12 \ months = 5.5 GB
$$

The actual size of a one month file is 5.4 GB which is much easier to work with as a personal computer could read all of this data in at once, and perform the math needed to compute EOFs. 

## Computing Standard Deviation and Climatology
Before computing EOFs climatology  and standard deviation are computed. These are computed for all 33 depths at once therefore each year has $$  N = 360 \times 180 \times 33 = 2,138,400 $$ data points. In python it’s simple to compute climatology(mean) and standard deviation:

```python
import warnings

# I expect to see RuntimeWarnings in this block
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    clim = np.mean(data_T, axis = 1)
sdev = np.std(data_T, axis = 1)
clim_sdev = np.column_stack((clim, sdev))

```
Note the argument “axis = 1”  in each function call for the N by Y data “data_T” tells each function to take the mean or standard deviation for that row. 

![Climatology]({{ site.url }}/assets/img/post1/clim_jan_depth5.png){: .center-image }

<center>Figure 1: Climatology computed using python with depth of 5m</center>

![Climatology]({{ site.url }}/assets/img/post1/clim_jan_depth600.png){: .center-image }

<center>Figure 2: Climatology computed using python with depth of 600m</center>


![StandardDev]({{ site.url }}/assets/img/post1/sdev_jan_depth5.png){: .center-image }

<center>Figure 3: Standard deviation computed using python with depth of 5m</center>

![StandardDev]({{ site.url }}/assets/img/post1/sdev_jan_depth600.png){: .center-image }

<center>Figure 4: Standard deviation computed using python with depth of 600m</center>

## Compute Anomalies 
Computing anomalies from this point is simple. Merely subtract each data point with their corresponding climatology:

```python 
anom = data - clim
```
![anomalies]({{ site.url }}/assets/css/img/greg_anom/Top Layer577_Reconstructed_Temp_Anomaly_Jan1998.png){: .center-image }

<center>Figure 5: Anomalies of Jan 1998 top layer</center>

![anomalies]({{ site.url }}/assets/img/post1/anom_jan1998_depth5.png){: .center-image }

<center>Figure 6: Anomalies of Jan 1998 top layer computed using python</center>


![anomalies]({{ site.url }}/assets/css/img/greg_anom/600m577_Reconstructed_Temp_Anomaly_Jan1998.png){: .center-image }

<center>Figure 7: Anomalies of Jan 1998 600m</center>


![anomalies]({{ site.url }}/assets/img/post1/anom_jan1998_depth600.png){: .center-image }

<center>Figure 8: Anomalies of Jan 1998 600m computed using python</center>

If we divide the anomalies by their respective standard deviation  then their standardized anomalies can be found:

```python
stnd_anom = anom/sdev
```

![anomalies]({{ site.url }}/assets/img/post1/std_anom_jan1998_depth600.png){: .center-image }

<center>Figure 9: Standardized anomalies of Jan 1998 600m computed using python</center>

Closer to the poles the grid boxes tend to get smaller. For this reason the standardized anomalies are multiplied by an area weight. This weight is based on latitude radian values.

$$ Cos(\phi \times \frac{\pi}{180}) $$

where %% \phi$$ is the lattitude values. 

from here we find weighted anomalies by:

$$ A_w = \frac{A}{\sigma} \times Cos(\phi \times \frac{\pi}{180}) $$

The weighted anomalies are what are used to compute covariance and then EOFs.

## Compute Covariance, Eigenvectors, and Eigenvalues

Because the data takes into account land, there are NaN values in the data. To get rid of this we take advantage of the fact that no matter what year it is land does not move. As a result if a row has a NaN value every column in that row also has a NaN value meaning that entire row is NaN. This makes it easy to know how many rows have values and how many rows do not have values. We can then fill a new matrix with only values and no NaNs.

Finding how many rows have values:
```python
na_rows = np.argwhere(np.isnan(anom) == False)
num_rows = []
count = 0
for i in range(1036304):
    num_rows.append(na_rows[i*54,0])
numrows = np.array(num_rows)
```

Inputing values into new matrix:
```python
new_anom = np.empty((numrows.shape[0],Y))
for i in range(numrows.shape[0]):
        new_anom[i,:] = weighted_A[numrows[i],:]
new_anom = np.mat(new_anom)
```

Covariance is computed as explained above:

```python
cov = (new_anom.transpose() * new_anom)/N
```

The eigenvalues and eigenvectors are found using:
```python
eigvals, eigvecs = la.eig(cov)
```

The figure below shows both the variance percentage and cumulative variance percentage. Variance shows the amount of the original data each eigenvalue explains. The cumulative variance reflects the amount of the original data is explained at that eigenvalue and the values before. 

![Scree]({{ site.url }}/assets/img/post1/scree_plot.png){: .center-image }

<center>Figure 10: Variance percentage and Cumulative variance percentage for Jan</center>

For the most part this plot shows behavior that is typical. Eigenvalues should decrease. They are sorted in order of importance. The only mode that is atypical is mode 10. For this reason the maximum amount of modes used in the reconstruction will be less than 10. 

## Computing EOFs for the Smaller Dataset
EOFs are computed by finding the eigenvectors of the temporal covariance matrix, multiplying eigenvectors by the anomalies, and dividing the magnitude of the multiplication. 

```python
EOF = np.empty((anom.shape[0],anom.shape[1]))
for i in range(anom.shape[1]):
    for j in range(anom.shape[0]): 
        EOF[j,i] = weighted_A[j,:] * eigvecs[:,i]
        
mag = np.zeros(Y)
for i in range(Y):
    mag[i] = np.linalg.norm(new_EOF[:,i])
    EOF[:,i] = EOF[:,i]/mag[i]
```

Where new_EOF is just the EOF matrix without NaN values. As a way to check that these EOFs are valid the magnitude of each EOF mode should be 1.

```python
# checking magnitude of EOF. Each should be 1
for i in range (54):
    EOF1 = np.mat(EOF[:,i])
    nan_flag = np.isnan(EOF1)
    EOF1_clean = EOF1[~nan_flag]
    print(np.linalg.norm(EOF1_clean))
```

These are the resulting EOFs computed from time covariance. 


![EOF]({{ site.url }}/assets/img/post1/EOF_jan_mode_0_depth_10.png){: .center-image }

<center>Figure 11: EOF mode 1 computed using time covariance </center>


![EOF]({{ site.url }}/assets/img/post1/EOF_jan_mode_1_depth_10.png){: .center-image }

<center>Figure 12: EOF mode 2 computed using time covariance </center>

![EOF]({{ site.url }}/assets/img/post1/EOF_jan_mode_2_depth_10.png){: .center-image }

<center>Figure 13: EOF mode 3 computed using time covariance </center>

![EOF]({{ site.url }}/assets/img/post1/EOF_jan_mode_3_depth_10.png){: .center-image }

<center>Figure 14: EOF mode 4 computed using time covariance </center>

These EOFs are consistent with the EOFs in Shen et al. 2017 and will be used for the reconstruction of climate data through multivariate regression.


