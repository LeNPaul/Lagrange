---
layout: post
title: "Computing EOFs Using Covariance in Time"
author: "Dani Lafarga"
categories: journal
tags: [documentation,sample]
image: phys_EOF1_Dec_depth_5500.gif
---


## Motivation
Earlier 2-dimensional oceanic reconstructions were often calculated for sea surface temperature (SST) due to the availability of data, and impor- tance of SST in the ocean’s direct influence on weather. Although the deep ocean temperature field has fewer observational data, the oceanic dynamics, such as equatorial upwelling, implies there is an existence of covariance among the temperatures of different layers. For this reason a 3- dimensional reconstruction with the inclusion of deep ocean temperatures is crucial. In order to optimize the reconstruction of the temperature field from surface to 2,000 meters depth, we have utilized temporal covariance, the NASA JPL ocean general circulation model (OGCM), and NOAA’s in situ observational data. Using machine learning, we can complete the 3-dimensional reconstruction at monthly time resolution with 26 layers. From this reconstruction we are able to quantitatively detect and visualize significant ocean dynamic features, such as the cold El Nin ̃o anomalies in deep ocean over the western Tropical Pacific, and equatorial upwelling.

## Background
### What is covariance?
Covariance measures how much two variables change. Typically in climatology covariance is considered between stations, grid boxes, or grid points. Covariance between two grid boxes i and j can be denoted as:

$$ \sum _{ij}$$

For N grid boxes covariance would be an N by N matrix:

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

Covariance is defined as:

$$[\sum_{ij}]_{N\times N} = \frac1 Y A_{N\times Y} A_{Y\times N}^T$$

Covariance in space can then be used to find empirical orthogonal functions using SVD. As you will see, because our data set is so large computing spatial covariance is not possible, and as an alternative temporal covariance is computed. Meaning, instead of computing the covariance between each grid box we are computing the covariance between the same grid box for different years given a specific month.


$$[\sum_{tt'}]_{Y\times Y} = \frac1 N \sum_{i=1}^N a(t)a(t') = \frac1 N A_{Y\times N}^T A_{N\times Y}$$

from this we can also compute emperical orthogonal functions.

### What are Empirical Orthogonal functions?
Emperical orthogonal functions are eigenvectors. Eigenvectors are vectors that point in the same direction as their corresponding matrix.

Consider the square covariance matrix $$[\sum_{ij}]_{N\times N}$$ and some vector  $$\vec{u}$$ which runs parallel to $$[\sum_{ij}]_{N\times N}$$. There is a scalar or eigenvalue $$\rho$$ which scales $$\vec{u}$$ such that:

$$[\sum _{ij}]_{N\times N}  \vec{u} = \rho \vec{u}$$
 
The first few eigenvectors of a large climate covariance matrix of climate data often represent some typical patterns of climate variability (Shen and Somerville 97). Usually EOFs are computed using singular value decomposition (SVD), but the method  used here is first finding covariance in time and then computing eigenvectors from that covariance matrix followed by multiplying the vectors to the anomaly matrix.

### Covariance in Time and EOFs 
Consider some data $$x_{it}$$ whose anomalies are $$ A_{N\times Y}$$ their spatial covariance is then:

$$ [\sum _{ij}]_{N\times N} = A_{N\times Y} A_{Y\times N}^T$$

and their covarience with respect to time is:

$$ [\sum_{tt'}]_{Y\times Y} = A_{Y\times N}^T A_{N\times Y}$$

In space there is some vector $$\vec{v}$$ that points in the same direction as $$ [C_{ij}]_{N\times N}$$ such that

$$ [\sum _{ij}]_{N\times N} \vec{u} = \rho \vec{u}$$

In time there is some other vector $$\vec{v}$$ that points in the same direction as $$[\sum _{tt'}]_{Y\times Y}$$ such that:

$$[\sum _{tt'}]_{Y\times Y}  \vec{v} = \lambda \vec{v}$$

Meaning $$\vec{u}$$ are the eigenvectors of  $$[\sum _{ij}]_{N\times N}$$ with $$\rho$$ being it's eigenvalues, and $$\vec{v}$$ are the eigenvectors of  $$[\sum _{tt'}]_{Y\times Y}$$ with $$\lambda$$ being it's eigenvalues. 

The problem we are trying to answer is how these things eigenvectors and eigenvalues relate? Above we defined covariance in time as
$$[\sum_{tt'}]_{Y\times Y} = \frac1 N A_{Y\times N}^T A_{N\times Y} $$. We can ignore the  1/N as this will only scale the unique vector and not change its direction. Therefore:

$$[\sum_{tt'}]_{Y\times Y} = A_{Y\times N}^T A_{N\times Y} $$

plugging this into its eigenvalue problem:

$$A_{Y\times N}^T A_{N\times Y} \vec{v} = \lambda \vec{v}$$

Multiply both sides by A:

$$A_{N\times Y} A_{Y\times N}^T A_{N\times Y} \vec{v} = \lambda A_{N\times Y}\vec{v}$$

because $$[\sum _{ij}]_{N\times N} = A_{N\times Y} A_{Y\times N}^T$$ we can redefine the equation above as:

$$[\sum _{ij}]_{N\times N} A_{N\times Y} \vec{v} = \lambda A_{N\times Y}\vec{v}$$

let $$\vec{w} = A_{N\times Y} \vec{v}$$ this gives:

$$[\sum _{ij}]_{N\times N} \vec{w} = \lambda \vec{w}$$

This is similar to:
$$ [\sum _{ij}]_{N\times N} \vec{u} = \rho \vec{u}$$

Comparing the two equations we can conclude: $$ \rho = \lambda$$ and $$ \vec{w} = A \vec{v}$$. $$ \vec{w}$$ does not equal to one so  it needs to be normalized. To do this  we divide it by its magnitude 

$$ EOF = \frac{A \vec{v}} {norm(A \vec{v})} $$

## The Data
The data used in the reconstruction of ocean temperatures is from JPL’s non-Boussinesq ocean general circulation model (OGCM). This data was initially collected on a $$1/4^\circ$$ by $$1/4^\circ$$ grid with ocean temperatures (in $$ ^\circ c$$) of 33 depths (in m) taken for 54 years. The data  is taken every 10 days making this quite a large problem for computing EOFs. The total amount of bytes needed to be read is:

$$1/4^\circ \times 1/4^\circ \times 32 \ layers = 1442 \times 698 \times 32 = 32,208,512\ entries $$


there are 8 bytes for each dataum therefore there is:

$$ 8 \ bytes \times 32,208,521  \ entries = 257MB \ per\ file$$

The actual amount of bytes per entry is 274MB. There are 37 files for each year as each file represents a 10 day block  with the last on representing 5 days, and this is done for 42 years so there are 1998 files:

$$ \frac{1\ file}{10\ days}\times \frac{365\ days}{year}  = \frac {37\ files}{years} \times 42 \ years = 1554 \ files$$


The total amount of bytes is:
$$ 1554 \ files \times 274MB = 425GB \times \ temperature \ and \ salinity $$
$$= 425GB \times 2 = 850GB$$


To model this data we start with a larger resolution and therefore a smaller amount of data. Consider instead computing EOFs for each month from 1950-2003 for 32 depths of the ocean. On a one degree by one degree grid this would mean:

$$ 1 ^\circ \times 1^\circ \times 32 \ layers\ = 360 \times 180 \times 32 = 2,073,600\ entries$$

If there are 54 years needed to be modeled and there are 4 bytes per datum then:

$$
2,073,600\ entries \times 4Bytes \times 54 \ years \times  = 447 GB
$$

The actual size of a one month file is 450 GB which is much easier to work with on a personal computer. We use this larger resolution to verify the method outlined above. This is compared to another method in python that uses SVD and spatial covariance to compute EOFs. For more information on that method see [here.](https://ajdawson.github.io/eofs/latest/api/eofs.standard.html) 

## Computing Standard Deviation and Climatology
Before computing EOFs climatology  and standard deviation are computed. These are computed for all 32 depths at once therefore each year has $$  N = 360 \times 180 \times 33 = 2,073,600 $$ data points. In python it’s simple to compute climatology(mean) and standard deviation:

```python
import warnings

# I expect to see RuntimeWarnings in this block

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    clim = np.nanmean(data_T, axis = 1)   # climatology  
    sdev = np.nanstd(data_T, axis = 1)    # standard deviation
clim_sdev = np.column_stack((clim, sdev)) # save this as one matrix 

```
Note the argument “axis = 1”  in each function call for the N by Y data tells each function to take the mean or standard deviation for that row. 

![Climatology]({{ site.url }}/assets/img/post1/clim_Dec_depth2000.gif){: .center-image }

<center>Figure 1: Climatology computed using python</center>

![StandardDev]({{ site.url }}/assets/img/post1/sdev_Dec_depth2000.gif){: .center-image }

<center>Figure 2: Standard deviation computed using python</center>


## Compute Anomalies 
Computing anomalies from this point is simple. Merely subtract each data point with their corresponding climatology:

```python 
anom = data - clim
```

![anomalies]({{ site.url }}/assets/img/post1/anom_jan1998_depth5.png){: .center-image }

<center>Figure 3: Anomalies of Jan 1998 top layer computed using python</center>


![anomalies]({{ site.url }}/assets/img/post1/anom_jan1998_depth600.png){: .center-image }

<center>Figure 4: Anomalies of Jan 1998 600m computed using python</center>

If we divide the anomalies by their respective standard deviation  then their standardized anomalies can be found:

```python
stnd_anom = anom/sdev
```

Closer to the poles the grid boxes tend to get smaller. For this reason the anomalies are multiplied by an area weight. This weight is based on latitude radian values.

$$ Cos(\phi \times \frac{\pi}{180}) $$

from here we find weighted anomalies by:

$$ A_w = \frac{A}{\sigma} \times Cos(\phi \times \frac{\pi}{180}) $$

The weighted anomalies are what are used to compute covariance and then EOFs.

this is how to find the area weights in python 
```python
# find lattitude and Longitude values
x = linspace(0, 360, 360)
y = linspace(-90, 90, 180)

xx, yy = meshgrid(x, y)

yy = yy.transpose()
y_temp = yy.flatten()

# area weight for lattitude values
area_w = np.cos(y_temp*math.pi/180)

# area weights for each depth depth
area_weight = []
for i in range(tot_depths):
    # first area weight is just 5
    if i == 0:
        area_weight.append(np.sqrt(5 * area_w)) # first depth thickness is just 0-5m
    else:
        area_weight.append( np.sqrt((depths[i] - depths[i - 1]) * area_w))
# Turning weights into one array
area_weight = np.mat(area_weight)
area_weight = area_weight.flatten()
area_weight = area_weight.transpose()

# Multiply weight with Anomalies
weighted_A = np.empty((N,Y)) * np.nan
weighted_A = np.multiply(anom , area_weight)
```

## Compute Covariance, Eigenvectors, and Eigenvalues

Because the data takes into account land, there are NaN values in the data. To get rid of this we take advantage of the fact that no matter what year it is land does not move. As a result if a row has a NaN value every column in that row also has a NaN value meaning that entire row is NaN. This makes it easy to know how many rows have values and how many rows do not have values. We can then fill a new matrix with only values and no NaNs.

Finding how many rows have values:
```python
na_rows = np.argwhere(np.isnan(weighted_A) == False)
new_N = round(na_rows.shape[0]/54)
```

Because entire rows have NaN the indicies are repeated for every column (Year). We get rid of repeats to consolidate array which will give the row index without NaN:
```python
num_rows = []
count = 0
for i in range(new_N):
    num_rows.append(na_rows[i*Y,0])
numrows = np.array(num_rows)
```

Create matrix with only values
```python
new_anom = np.empty((new_N,Y)) * np.nan
for i in range(new_N):
        new_anom[i,:] = weighted_A[numrows[i],:]
new_anom = np.mat(new_anom)
new_anom.shape
```

Covariance is computed as explained above:

```python
cov = (new_anom.transpose() * new_anom)/N
```

The eigenvalues and eigenvectors are found using:
```python
eigvals, eigvecs = la.eig(cov)
```

These eigenvalues and eigenvectors ARE NOT sorted. Its easy to sort eigenvalues low to higher values using a python function which keeps track of the correct indexing:
```python
# find correct index
eig_index = np.argsort(eigvals)
eig_index = np.flip(eig_index)

# sort evals
new_eigvals = eigvals[eig_index[:]]
eigvals = new_eigvals

# sort evecs
new_eigvecs = eigvecs[:,eig_index]
eigvecs = new_eigvecs 
```


The figure below shows both the variance percentage and cumulative variance percentage. Variance shows the amount of the original data each eigenvalue explains. The cumulative variance reflects the amount of the original data is explained at that eigenvalue and the values before. 

![Scree]({{ site.url }}/assets/img/post1/Dec_screePlot.png){: .center-image }

<center>Figure 10: Variance percentage and Cumulative variance percentage for Dec</center>

This scree plot shows the percentage variance of each mode and up to how much information cumulatively we will have at a specific mode. The latter will come in handy during the multivariate  regression in telling us how many modes we would want to use for a large percent of information. The percentage variance will tell us how important each mode is.

## Computing EOFs for the Smaller Dataset
EOFs are computed by finding the eigenvectors of the temporal covariance matrix, multiplying eigenvectors by the anomalies, and dividing the magnitude of the multiplication. 

```python
ev = eigvecs.T
EOFs = []
for j in range(Y):
    EOFs.append(np.matmul(weighted_A , ev[j].T))
    
EOF = np.array(EOFs)
EOF = np.squeeze(EOF)
EOF = EOF.T

mag = np.zeros(Y)
for i in range(Y):
    mag[i] = np.linalg.norm(new_EOF[:,i])
    EOF[:,i] = EOF[:,i]/mag[i]
    
```

Where new_EOF is just the EOF matrix without NaN values. As a way to check that these EOFs are valid the magnitude of each EOF mode should be 1.

```python
# checking magnitude of EOF. If magnitude is not one error will be raised
for col_i in range(54):
    is_one = np.linalg.norm(new_EOF[:,col_i])
    if not np.isclose(is_one, 1):
        raise ValueError(f"EOF {col_i} is not one")
```

It is also important to check othogonality.

```python
# Check to make sure vectors are orthogonal
n, m = new_EOF.shape
for col_i in range(m):
    for col_j in range(m):
        if col_i < col_j:  # use strictly less than because we don't want to consider a column with itself, and the dot product is commutable so order doesn't matter
            is_orthogonal = np.dot(new_EOF[:, col_i], new_EOF[:, col_j])
            if not np.isclose(is_orthogonal, 0) and col_j != 53:
                pprint(f"EOF Mode {col_i} and EOF Mode {col_j} are not orthogonal.")
                pprint(np.dot(new_EOF[:, col_i], new_EOF[:, col_j]))
```

finally to find physical EOFs divide by the area weight 

```python
EOF = EOF/area_weight[:,0]
```

These are the resulting EOFs computed from time covariance. 


![EOF]({{ site.url }}/assets/img/post1/phys_EOF1_Dec_depth_5500.gif){: .center-image }

<center>Figure 11: EOF mode 1 computed using time covariance </center>

Surprisingly the first mode does not show the most prominent characteristic to be El Nino. This is what previous calculations done in 2D showed. The first mode has the most amount of percent variance being at around 30%. Meaning most variance is found in the first mode. We would expect then for El Nino to appear within this mode but instead we see equitorial upwelling. This is when deep ocean waters rise up and push surface waters outwards. The next mode starts to show the split between cold and warmer waters at the oceans  surface as expected from El Nino. 

![EOF]({{ site.url }}/assets/img/post1/phys_EOF2_Dec_depth_5500.gif){: .center-image }

<center>Figure 12: EOF mode 2 computed using time covariance </center>





