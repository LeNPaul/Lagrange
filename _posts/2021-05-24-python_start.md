---
layout: post
title: "Reconstructing Climate data Using Python"
author: "Dani Lafarga"
categories: journal
tags: [documentation,sample]
---
# Motivation
Sea surface temperature (SST) is studied more often than deep sea temperatures. More recently there has been more interest in deep ocean temperatures, but there has been a limitation to research due to the lack of measurement in those areas. The objective of this NERTO research is to make a high-resolution reconstruction of deep ocean temperatures of depths up to 5,500 meters at ¼ degree spatial resolution and 5-day time resolution with 33 layers for 26 years.

# Background
## What is covariance?
Covariance measures how much two variables change. Typically in climatology covariance is considered between stations, grid boxes, or grid points. Covariance between two stations i and j can be denoted as:

$$ \sum _{ij}$$

For N stations covariance would be an N by N matrix:

$$ [\sum _{ij}]_{N\times N}$$

Consider data X that consists of N stations and Y time steps:
$$ X = [x_{it}]_{N\times Y}$$

Covarance can then be computed by first computing anomalies:
$$ A_{N \times Y} = [a_{it}]_{N\times Y} =  [x_{it} - \bar x_{i}]_{N\times Y}$$

Where:
$$\bar x = \frac1 Y \sum_{t=1} ^Y x_{it}$$
and recall:
$$ i = 1,2...N$$

This is the mean of the data or more formally climatology.

Covariance is computed by:
$$[\sum_{ij}]_{N\times N} = \frac1 Y A_{N\times Y} A_{Y\times N}^T$$

for this case becuase the data set is so large the  covariance is computed in terms of time rather than in terms of space. Instead of computing the covariance for each station we are computing  the covariance for every year given a specific month. 

$$[\sum_{tt'}]_{Y\times Y} = \frac1 N \sum_{i=1}^N a(t)a(t') = \frac1 N A_{Y\times N} A_{N\times Y}^T$$

from this we can compute emperical orthogonal functions.

## What are Empirical Orthogonal functions?
Emperical orthogonal function come from eigenvectors. Consider the square covariance matrix $$[\sum_{tt'}]_{Y\times Y} $$ and some vector  w which runs parallel to $$[\sum_{tt'}]_{Y\times Y} $$ . There is a scalar or eigenvalue $$\lambda$$ which scales w such that:

$$[\sum _{tt'}]_{Y\times Y} \times w = \lambda \times w$$
 
The first few eigenvectors of a large climate covariance matrix of climate data often represent some typical patterns of climate variability (Shen and Somerville 97). Usually EOFs are computed using singular value decomposition (SVD), but the method  used here is first finding covariance in time and then computing eigenvectors from that covariance matrix followed by multiplying the vectors to the anomaly matrix.

For example the reconstruction of climate data using the first eigen vector would look like
$$A \times w_1$$

#  Week 1
The first step taken to model this data was to start with a larger resolution and therefore a smaller amount of data. Consider then computing emperical orthogonal functions (EOFs) for each month from 1950-2003 for 33 depths of the ocean. On a one degree by one degree grid this would mean:

$$ 1 ^\circ \times 1^\circ \times 33 \ layers\ = 360 \times 180 \times 33 = 2,138,400\ Bytes\ of\ data\ per\ month$$


If there are 54 years needed to be modeled and there are 4 bytes per datum then:

$$
2MB \times 4B \times 54 \ years \times 12 \ months = 5.5 GB
$$

The actual size of a one month file is 5.4 GB which is much easier to work with as a personal computer could read all of this data in at once, and perform the math needed to compute EOFs via SVD. This is not the  case for a $$1/4^\circ$$ by $$1/4^\circ$$ grid as there would be a lot more data needed to be read in to compute the EOFs. In that case the data is taken for every 5 days as opposed to every month. The total amount of bytes needed to be read is:

$$1/4^\circ \times 1/4^\circ \times 33 \ layers \  \times 5 \ days$$

$$ = 1442 \times 698 \times 33 = 33,215,028 \ entries$$

there are 8 bytes for each dataum therefore there is:

$$ 8 \times 33,215,028 = 265MB \ per\ entry$$

The actual amount of bytes per entry is 274MB. There are 73 files for each year as each file represents a 5 day block and this is done for 26 years so there are 1901 files:

$$ \frac{365\ days}{year} \times \frac{1\ file}{5\ days} = \frac {73\ files}{years} \times 26 \ years = 1901 \ files$$
The total amount of bytes is:

$$ 1901 \ files \times 274MB = 521GB \times \ temperature \ and \ salinity = 521 \times 2 = 1.04TB \times 3 = 3TB$$

As mentioned above EOFs are computed by finding the eigenvectors of the temporal covariance matrix. In this case the covariance matrix is computed on a 1 degree grid for 54 years therefor N = 54. Eigenvectors should be of dimension 1 by N and there should be N eigenvectors for each year. 

Prior to computing EOFs the climatology for the smaller data is computed along with the standard deviation. The following figures are the results of computing the climatology, standard deviation, anomalies, and EOFs:

The files that wer impoted were matlab files that were then turned into a matrix:

NOTE: use ```python os.chdir``` to change the directory  to the directory the files are in

```python
data = {}
for file in os.listdir():
    if ".mat" in file:
        data[file] = sc.loadmat(file)["data"][:,:,16].flatten()
data = list(data.values())
data = np.mat(data)
```
The line ```python data[file] = sc.loadmat(file)["data"][:,:,depth].flatten()``` will load each multidimensional matrix named data into a dictionary where every key is the name of each 360 x 180  x 33 matrix. 

To compute climatology and standard deviation:

```python
clim_sdev  = np.empty(shape = [len(data),2])

import warnings

# I expect to see RuntimeWarnings in this block
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    clim = np.nanmean(data, axis = 1)
sdev = np.std(data, axis = 1)
clim_sdev = np.column_stack((clim, sdev))
```
The plot is identical to the reference graph in figure 1.


![Climatology]({{ site.url }}/assets/css/img/clim/clim_jan600.png){: .center-image }

<center>Figure 1: Climatology computed using python with depth of 600</center>


![standard deviation]({{ site.url }}/assets/css/img/std_dev/standarddev_jan600.png){: .center-image }

<center>Figure 2: Standard deviation computed using python with depth of 600</center>

Computing anomalies from this point is simple. Merely subtract each data point with their corresponding climatology:

```python 
anom = data - clim
```
![anomalies]({{ site.url }}/assets/css/img/greg_anom/Top Layer577_Reconstructed_Temp_Anomaly_Jan1998.png){: .center-image }

<center>Figure 3: Anomalies of jan 1998 top layer</center>

![anomalies]({{ site.url }}/assets/css/img/anom/depth5/anom_jan1998_depth5.png){: .center-image }

<center>Figure 4: Anomalies of jan 1998 top layer computed using python</center>

![anomalies]({{ site.url }}/assets/css/img/greg_anom/600m577_Reconstructed_Temp_Anomaly_Jan1998.png){: .center-image }

<center>Figure 5: Anomalies of jan 1998 600m</center>

![anomalies]({{ site.url }}/assets/css/img/anom/depth600/anom_jan1998.png){: .center-image }

<center>Figure 6: Anomalies of jan 1998 600m computed using python</center>

At this point covariance needs to be computed. To do so NaN need to be romoved from the data using:
```python
new_data = []
for i in range(0,data.shape[0]):
    temp = []
    for j in range(0, data.shape[1]): 
        if ~np.isnan(data[i,j]):
            temp.append(data[i,j])
    new_data.append(temp)
 ```
Followed by the simple command:

```python
cov = np.cov(new_data)
```
 then compute eigenvalues and vectors using:
 ```python
 eigvals, eigvecs = la.eig(cov)
 ```
 Compute EOFs by multiplying each eigenvector to the anomalies:
 ```python
 EOF = []
for j in range(anom.shape[1]): 
    temp = []
    for i in range(anom.shape[0]):
        temp = anom[:,j] * eigvecs[i,:]
    EOF.append(temp)
 ```
 and plot using contourf. 
 
![EOF]({{ site.url }}/assets/css/img/greg_EOF/1modes_5m-5500m_577_Reconstructed_Temp_Anomaly_Jan1998_Bn.png){: .center-image }

<center>Figure 7: EOF mode 1</center>

![EOF]({{ site.url }}/assets/css/img/EOF/EOF_jan1depth of10.png){: .center-image }

<center>Figure 8: EOF mode 1 computed using python</center>

![EOF]({{ site.url }}/assets/css/img/greg_EOF/2modes_5m-5500m_577_Reconstructed_Temp_Anomaly_Jan1998_Bn.png){: .center-image }

<center>Figure 9: EOF mode 2</center>

![EOF]({{ site.url }}/assets/css/img/EOF/EOF_jan2depth of10.png){: .center-image }

<center>Figure 10: EOF mode 2 computed using python</center>

![EOF]({{ site.url }}/assets/css/img/greg_EOF/3modes_5m-5500m_577_Reconstructed_Temp_Anomaly_Jan1998_Bn.png){: .center-image }

<center>Figure 11: EOF mode 3</center>

![EOF]({{ site.url }}/assets/css/img/EOF/EOF_jan3depth of10.png){: .center-image }

<center>Figure 12: EOF mode 3 computed using python</center>

![EOF]({{ site.url }}/assets/css/img/greg_EOF/4modes_5m-5500m_577_Reconstructed_Temp_Anomaly_Jan1998_Bn.png){: .center-image }

<center>Figure 13: EOF mode 4</center>

![EOF]({{ site.url }}/assets/css/img/EOF/EOF_jan4depth of10.png){: .center-image }

<center>Figure 14: EOF mode 4 computed using python</center>



