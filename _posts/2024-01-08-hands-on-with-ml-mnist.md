---
layout: post
title: "Getting Hands On With Machine Learning: MNIST"
author: "Henry Chang"
categories: journal
tags: [deeplearning]
image: mnist/mnist_22_0.png
---



One of the most commonly used datasets in ML research has been the MNIST dataset. It's a small dataset containing 60,000 28x28 greyscale images of handwritten digits and their labels, with 10 distinct classes for the digits from 0-9. It's often used as a "Hello World" example for machine learning. 

In this post, we go from doing some exploratory data analysis, through fully-connected models, up to training our first CNN model and achieving >99% classification accuracy on the dataset. Along the way, we write our own Learner class from scratch and use it to group together model, loss, optimizer, training and validation datasets.

This work was inspired by [this notebook](https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb) from the Fast.ai course, Practical Deep Learning for Coders.

You can choose to follow along directly in Colab, or read the summary below.
<a target="_blank" href="https://colab.research.google.com/github/henryjchang/dl-notebooks/blob/main/mnist.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Import torchvision MNIST

We download the MNIST dataset from torchvision. As part of the download, we can specify any transformations to make to the data inputs or to the data labels. Here, we only transform the data inputs. To be able to properly do machine learning with images, we need to turn them from PIL images to PyTorch Tensors. 
```python
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import Subset, DataLoader
import torch

train_dataset = MNIST(root='mnist', train=True, download=True, transform=T.ToTensor())
test_dataset = MNIST(root='mnist', train=False, download=True, transform=T.ToTensor())
```


### A first look at the data

Each image is a 3D tensor with a single channel (first dimension) and height and width of 28. We can get the image and label of the *i*th sample of the dataset like so:


```python
int i = 0
image = train_dataset[i][0]
label = train_dataset[i][1]
```



It's good practice to normalize input data to make the optimization landscape smoother so that it takes fewer iterations to converge to a good minimum. With pandas DataFrames, we can easily notice all the values have already been normalized to be between 0 and 1, with 1 corresponding with "black" and 0 with "white."


```python
import pandas as pd

df = pd.DataFrame(torch.squeeze(train_dataset[0][0]))
df.style.set_properties(**{'font-size':'5pt'}) \
    .background_gradient('Greys').format(precision=2)
```




<style type="text/css">
#T_0aae8_row0_col0, #T_0aae8_row0_col1, #T_0aae8_row0_col2, #T_0aae8_row0_col3, #T_0aae8_row0_col4, #T_0aae8_row0_col5, #T_0aae8_row0_col6, #T_0aae8_row0_col7, #T_0aae8_row0_col8, #T_0aae8_row0_col9, #T_0aae8_row0_col10, #T_0aae8_row0_col11, #T_0aae8_row0_col12, #T_0aae8_row0_col13, #T_0aae8_row0_col14, #T_0aae8_row0_col15, #T_0aae8_row0_col16, #T_0aae8_row0_col17, #T_0aae8_row0_col18, #T_0aae8_row0_col19, #T_0aae8_row0_col20, #T_0aae8_row0_col21, #T_0aae8_row0_col22, #T_0aae8_row0_col23, #T_0aae8_row0_col24, #T_0aae8_row0_col25, #T_0aae8_row0_col26, #T_0aae8_row0_col27, #T_0aae8_row1_col0, #T_0aae8_row1_col1, #T_0aae8_row1_col2, #T_0aae8_row1_col3, #T_0aae8_row1_col4, #T_0aae8_row1_col5, #T_0aae8_row1_col6, #T_0aae8_row1_col7, #T_0aae8_row1_col8, #T_0aae8_row1_col9, #T_0aae8_row1_col10, #T_0aae8_row1_col11, #T_0aae8_row1_col12, #T_0aae8_row1_col13, #T_0aae8_row1_col14, #T_0aae8_row1_col15, #T_0aae8_row1_col16, #T_0aae8_row1_col17, #T_0aae8_row1_col18, #T_0aae8_row1_col19, #T_0aae8_row1_col20, #T_0aae8_row1_col21, #T_0aae8_row1_col22, #T_0aae8_row1_col23, #T_0aae8_row1_col24, #T_0aae8_row1_col25, #T_0aae8_row1_col26, #T_0aae8_row1_col27, #T_0aae8_row2_col0, #T_0aae8_row2_col1, #T_0aae8_row2_col2, #T_0aae8_row2_col3, #T_0aae8_row2_col4, #T_0aae8_row2_col5, #T_0aae8_row2_col6, #T_0aae8_row2_col7, #T_0aae8_row2_col8, #T_0aae8_row2_col9, #T_0aae8_row2_col10, #T_0aae8_row2_col11, #T_0aae8_row2_col12, #T_0aae8_row2_col13, #T_0aae8_row2_col14, #T_0aae8_row2_col15, #T_0aae8_row2_col16, #T_0aae8_row2_col17, #T_0aae8_row2_col18, #T_0aae8_row2_col19, #T_0aae8_row2_col20, #T_0aae8_row2_col21, #T_0aae8_row2_col22, #T_0aae8_row2_col23, #T_0aae8_row2_col24, #T_0aae8_row2_col25, #T_0aae8_row2_col26, #T_0aae8_row2_col27, #T_0aae8_row3_col0, #T_0aae8_row3_col1, #T_0aae8_row3_col2, #T_0aae8_row3_col3, #T_0aae8_row3_col4, #T_0aae8_row3_col5, #T_0aae8_row3_col6, #T_0aae8_row3_col7, #T_0aae8_row3_col8, #T_0aae8_row3_col9, #T_0aae8_row3_col10, #T_0aae8_row3_col11, #T_0aae8_row3_col12, #T_0aae8_row3_col13, #T_0aae8_row3_col14, #T_0aae8_row3_col15, #T_0aae8_row3_col16, #T_0aae8_row3_col17, #T_0aae8_row3_col18, #T_0aae8_row3_col19, #T_0aae8_row3_col20, #T_0aae8_row3_col21, #T_0aae8_row3_col22, #T_0aae8_row3_col23, #T_0aae8_row3_col24, #T_0aae8_row3_col25, #T_0aae8_row3_col26, #T_0aae8_row3_col27, #T_0aae8_row4_col0, #T_0aae8_row4_col1, #T_0aae8_row4_col2, #T_0aae8_row4_col3, #T_0aae8_row4_col4, #T_0aae8_row4_col5, #T_0aae8_row4_col6, #T_0aae8_row4_col7, #T_0aae8_row4_col8, #T_0aae8_row4_col9, #T_0aae8_row4_col10, #T_0aae8_row4_col11, #T_0aae8_row4_col12, #T_0aae8_row4_col13, #T_0aae8_row4_col14, #T_0aae8_row4_col15, #T_0aae8_row4_col16, #T_0aae8_row4_col17, #T_0aae8_row4_col18, #T_0aae8_row4_col19, #T_0aae8_row4_col20, #T_0aae8_row4_col21, #T_0aae8_row4_col22, #T_0aae8_row4_col23, #T_0aae8_row4_col24, #T_0aae8_row4_col25, #T_0aae8_row4_col26, #T_0aae8_row4_col27, #T_0aae8_row5_col0, #T_0aae8_row5_col1, #T_0aae8_row5_col2, #T_0aae8_row5_col3, #T_0aae8_row5_col4, #T_0aae8_row5_col5, #T_0aae8_row5_col6, #T_0aae8_row5_col7, #T_0aae8_row5_col8, #T_0aae8_row5_col9, #T_0aae8_row5_col10, #T_0aae8_row5_col11, #T_0aae8_row5_col24, #T_0aae8_row5_col25, #T_0aae8_row5_col26, #T_0aae8_row5_col27, #T_0aae8_row6_col0, #T_0aae8_row6_col1, #T_0aae8_row6_col2, #T_0aae8_row6_col3, #T_0aae8_row6_col4, #T_0aae8_row6_col5, #T_0aae8_row6_col6, #T_0aae8_row6_col7, #T_0aae8_row6_col24, #T_0aae8_row6_col25, #T_0aae8_row6_col26, #T_0aae8_row6_col27, #T_0aae8_row7_col0, #T_0aae8_row7_col1, #T_0aae8_row7_col2, #T_0aae8_row7_col3, #T_0aae8_row7_col4, #T_0aae8_row7_col5, #T_0aae8_row7_col6, #T_0aae8_row7_col23, #T_0aae8_row7_col24, #T_0aae8_row7_col25, #T_0aae8_row7_col26, #T_0aae8_row7_col27, #T_0aae8_row8_col0, #T_0aae8_row8_col1, #T_0aae8_row8_col2, #T_0aae8_row8_col3, #T_0aae8_row8_col4, #T_0aae8_row8_col5, #T_0aae8_row8_col6, #T_0aae8_row8_col18, #T_0aae8_row8_col19, #T_0aae8_row8_col20, #T_0aae8_row8_col21, #T_0aae8_row8_col22, #T_0aae8_row8_col23, #T_0aae8_row8_col24, #T_0aae8_row8_col25, #T_0aae8_row8_col26, #T_0aae8_row8_col27, #T_0aae8_row9_col0, #T_0aae8_row9_col1, #T_0aae8_row9_col2, #T_0aae8_row9_col3, #T_0aae8_row9_col4, #T_0aae8_row9_col5, #T_0aae8_row9_col6, #T_0aae8_row9_col7, #T_0aae8_row9_col15, #T_0aae8_row9_col18, #T_0aae8_row9_col19, #T_0aae8_row9_col20, #T_0aae8_row9_col21, #T_0aae8_row9_col22, #T_0aae8_row9_col23, #T_0aae8_row9_col24, #T_0aae8_row9_col25, #T_0aae8_row9_col26, #T_0aae8_row9_col27, #T_0aae8_row10_col0, #T_0aae8_row10_col1, #T_0aae8_row10_col2, #T_0aae8_row10_col3, #T_0aae8_row10_col4, #T_0aae8_row10_col5, #T_0aae8_row10_col6, #T_0aae8_row10_col7, #T_0aae8_row10_col8, #T_0aae8_row10_col10, #T_0aae8_row10_col14, #T_0aae8_row10_col15, #T_0aae8_row10_col16, #T_0aae8_row10_col17, #T_0aae8_row10_col18, #T_0aae8_row10_col19, #T_0aae8_row10_col20, #T_0aae8_row10_col21, #T_0aae8_row10_col22, #T_0aae8_row10_col23, #T_0aae8_row10_col24, #T_0aae8_row10_col25, #T_0aae8_row10_col26, #T_0aae8_row10_col27, #T_0aae8_row11_col0, #T_0aae8_row11_col1, #T_0aae8_row11_col2, #T_0aae8_row11_col3, #T_0aae8_row11_col4, #T_0aae8_row11_col5, #T_0aae8_row11_col6, #T_0aae8_row11_col7, #T_0aae8_row11_col8, #T_0aae8_row11_col9, #T_0aae8_row11_col10, #T_0aae8_row11_col15, #T_0aae8_row11_col16, #T_0aae8_row11_col17, #T_0aae8_row11_col18, #T_0aae8_row11_col19, #T_0aae8_row11_col20, #T_0aae8_row11_col21, #T_0aae8_row11_col22, #T_0aae8_row11_col23, #T_0aae8_row11_col24, #T_0aae8_row11_col25, #T_0aae8_row11_col26, #T_0aae8_row11_col27, #T_0aae8_row12_col0, #T_0aae8_row12_col1, #T_0aae8_row12_col2, #T_0aae8_row12_col3, #T_0aae8_row12_col4, #T_0aae8_row12_col5, #T_0aae8_row12_col6, #T_0aae8_row12_col7, #T_0aae8_row12_col8, #T_0aae8_row12_col9, #T_0aae8_row12_col10, #T_0aae8_row12_col15, #T_0aae8_row12_col16, #T_0aae8_row12_col17, #T_0aae8_row12_col18, #T_0aae8_row12_col19, #T_0aae8_row12_col20, #T_0aae8_row12_col21, #T_0aae8_row12_col22, #T_0aae8_row12_col23, #T_0aae8_row12_col24, #T_0aae8_row12_col25, #T_0aae8_row12_col26, #T_0aae8_row12_col27, #T_0aae8_row13_col0, #T_0aae8_row13_col1, #T_0aae8_row13_col2, #T_0aae8_row13_col3, #T_0aae8_row13_col4, #T_0aae8_row13_col5, #T_0aae8_row13_col6, #T_0aae8_row13_col7, #T_0aae8_row13_col8, #T_0aae8_row13_col9, #T_0aae8_row13_col10, #T_0aae8_row13_col11, #T_0aae8_row13_col17, #T_0aae8_row13_col18, #T_0aae8_row13_col19, #T_0aae8_row13_col20, #T_0aae8_row13_col21, #T_0aae8_row13_col22, #T_0aae8_row13_col23, #T_0aae8_row13_col24, #T_0aae8_row13_col25, #T_0aae8_row13_col26, #T_0aae8_row13_col27, #T_0aae8_row14_col0, #T_0aae8_row14_col1, #T_0aae8_row14_col2, #T_0aae8_row14_col3, #T_0aae8_row14_col4, #T_0aae8_row14_col5, #T_0aae8_row14_col6, #T_0aae8_row14_col7, #T_0aae8_row14_col8, #T_0aae8_row14_col9, #T_0aae8_row14_col10, #T_0aae8_row14_col11, #T_0aae8_row14_col12, #T_0aae8_row14_col19, #T_0aae8_row14_col20, #T_0aae8_row14_col21, #T_0aae8_row14_col22, #T_0aae8_row14_col23, #T_0aae8_row14_col24, #T_0aae8_row14_col25, #T_0aae8_row14_col26, #T_0aae8_row14_col27, #T_0aae8_row15_col0, #T_0aae8_row15_col1, #T_0aae8_row15_col2, #T_0aae8_row15_col3, #T_0aae8_row15_col4, #T_0aae8_row15_col5, #T_0aae8_row15_col6, #T_0aae8_row15_col7, #T_0aae8_row15_col8, #T_0aae8_row15_col9, #T_0aae8_row15_col10, #T_0aae8_row15_col11, #T_0aae8_row15_col12, #T_0aae8_row15_col13, #T_0aae8_row15_col20, #T_0aae8_row15_col21, #T_0aae8_row15_col22, #T_0aae8_row15_col23, #T_0aae8_row15_col24, #T_0aae8_row15_col25, #T_0aae8_row15_col26, #T_0aae8_row15_col27, #T_0aae8_row16_col0, #T_0aae8_row16_col1, #T_0aae8_row16_col2, #T_0aae8_row16_col3, #T_0aae8_row16_col4, #T_0aae8_row16_col5, #T_0aae8_row16_col6, #T_0aae8_row16_col7, #T_0aae8_row16_col8, #T_0aae8_row16_col9, #T_0aae8_row16_col10, #T_0aae8_row16_col11, #T_0aae8_row16_col12, #T_0aae8_row16_col13, #T_0aae8_row16_col14, #T_0aae8_row16_col20, #T_0aae8_row16_col21, #T_0aae8_row16_col22, #T_0aae8_row16_col23, #T_0aae8_row16_col24, #T_0aae8_row16_col25, #T_0aae8_row16_col26, #T_0aae8_row16_col27, #T_0aae8_row17_col0, #T_0aae8_row17_col1, #T_0aae8_row17_col2, #T_0aae8_row17_col3, #T_0aae8_row17_col4, #T_0aae8_row17_col5, #T_0aae8_row17_col6, #T_0aae8_row17_col7, #T_0aae8_row17_col8, #T_0aae8_row17_col9, #T_0aae8_row17_col10, #T_0aae8_row17_col11, #T_0aae8_row17_col12, #T_0aae8_row17_col13, #T_0aae8_row17_col14, #T_0aae8_row17_col15, #T_0aae8_row17_col16, #T_0aae8_row17_col21, #T_0aae8_row17_col22, #T_0aae8_row17_col23, #T_0aae8_row17_col24, #T_0aae8_row17_col25, #T_0aae8_row17_col26, #T_0aae8_row17_col27, #T_0aae8_row18_col0, #T_0aae8_row18_col1, #T_0aae8_row18_col2, #T_0aae8_row18_col3, #T_0aae8_row18_col4, #T_0aae8_row18_col5, #T_0aae8_row18_col6, #T_0aae8_row18_col7, #T_0aae8_row18_col8, #T_0aae8_row18_col9, #T_0aae8_row18_col10, #T_0aae8_row18_col11, #T_0aae8_row18_col12, #T_0aae8_row18_col13, #T_0aae8_row18_col21, #T_0aae8_row18_col22, #T_0aae8_row18_col23, #T_0aae8_row18_col24, #T_0aae8_row18_col25, #T_0aae8_row18_col26, #T_0aae8_row18_col27, #T_0aae8_row19_col0, #T_0aae8_row19_col1, #T_0aae8_row19_col2, #T_0aae8_row19_col3, #T_0aae8_row19_col4, #T_0aae8_row19_col5, #T_0aae8_row19_col6, #T_0aae8_row19_col7, #T_0aae8_row19_col8, #T_0aae8_row19_col9, #T_0aae8_row19_col10, #T_0aae8_row19_col11, #T_0aae8_row19_col20, #T_0aae8_row19_col21, #T_0aae8_row19_col22, #T_0aae8_row19_col23, #T_0aae8_row19_col24, #T_0aae8_row19_col25, #T_0aae8_row19_col26, #T_0aae8_row19_col27, #T_0aae8_row20_col0, #T_0aae8_row20_col1, #T_0aae8_row20_col2, #T_0aae8_row20_col3, #T_0aae8_row20_col4, #T_0aae8_row20_col5, #T_0aae8_row20_col6, #T_0aae8_row20_col7, #T_0aae8_row20_col8, #T_0aae8_row20_col9, #T_0aae8_row20_col19, #T_0aae8_row20_col20, #T_0aae8_row20_col21, #T_0aae8_row20_col22, #T_0aae8_row20_col23, #T_0aae8_row20_col24, #T_0aae8_row20_col25, #T_0aae8_row20_col26, #T_0aae8_row20_col27, #T_0aae8_row21_col0, #T_0aae8_row21_col1, #T_0aae8_row21_col2, #T_0aae8_row21_col3, #T_0aae8_row21_col4, #T_0aae8_row21_col5, #T_0aae8_row21_col6, #T_0aae8_row21_col7, #T_0aae8_row21_col18, #T_0aae8_row21_col19, #T_0aae8_row21_col20, #T_0aae8_row21_col21, #T_0aae8_row21_col22, #T_0aae8_row21_col23, #T_0aae8_row21_col24, #T_0aae8_row21_col25, #T_0aae8_row21_col26, #T_0aae8_row21_col27, #T_0aae8_row22_col0, #T_0aae8_row22_col1, #T_0aae8_row22_col2, #T_0aae8_row22_col3, #T_0aae8_row22_col4, #T_0aae8_row22_col5, #T_0aae8_row22_col16, #T_0aae8_row22_col17, #T_0aae8_row22_col18, #T_0aae8_row22_col19, #T_0aae8_row22_col20, #T_0aae8_row22_col21, #T_0aae8_row22_col22, #T_0aae8_row22_col23, #T_0aae8_row22_col24, #T_0aae8_row22_col25, #T_0aae8_row22_col26, #T_0aae8_row22_col27, #T_0aae8_row23_col0, #T_0aae8_row23_col1, #T_0aae8_row23_col2, #T_0aae8_row23_col3, #T_0aae8_row23_col14, #T_0aae8_row23_col15, #T_0aae8_row23_col16, #T_0aae8_row23_col17, #T_0aae8_row23_col18, #T_0aae8_row23_col19, #T_0aae8_row23_col20, #T_0aae8_row23_col21, #T_0aae8_row23_col22, #T_0aae8_row23_col23, #T_0aae8_row23_col24, #T_0aae8_row23_col25, #T_0aae8_row23_col26, #T_0aae8_row23_col27, #T_0aae8_row24_col0, #T_0aae8_row24_col1, #T_0aae8_row24_col2, #T_0aae8_row24_col3, #T_0aae8_row24_col12, #T_0aae8_row24_col13, #T_0aae8_row24_col14, #T_0aae8_row24_col15, #T_0aae8_row24_col16, #T_0aae8_row24_col17, #T_0aae8_row24_col18, #T_0aae8_row24_col19, #T_0aae8_row24_col20, #T_0aae8_row24_col21, #T_0aae8_row24_col22, #T_0aae8_row24_col23, #T_0aae8_row24_col24, #T_0aae8_row24_col25, #T_0aae8_row24_col26, #T_0aae8_row24_col27, #T_0aae8_row25_col0, #T_0aae8_row25_col1, #T_0aae8_row25_col2, #T_0aae8_row25_col3, #T_0aae8_row25_col4, #T_0aae8_row25_col5, #T_0aae8_row25_col6, #T_0aae8_row25_col7, #T_0aae8_row25_col8, #T_0aae8_row25_col9, #T_0aae8_row25_col10, #T_0aae8_row25_col11, #T_0aae8_row25_col12, #T_0aae8_row25_col13, #T_0aae8_row25_col14, #T_0aae8_row25_col15, #T_0aae8_row25_col16, #T_0aae8_row25_col17, #T_0aae8_row25_col18, #T_0aae8_row25_col19, #T_0aae8_row25_col20, #T_0aae8_row25_col21, #T_0aae8_row25_col22, #T_0aae8_row25_col23, #T_0aae8_row25_col24, #T_0aae8_row25_col25, #T_0aae8_row25_col26, #T_0aae8_row25_col27, #T_0aae8_row26_col0, #T_0aae8_row26_col1, #T_0aae8_row26_col2, #T_0aae8_row26_col3, #T_0aae8_row26_col4, #T_0aae8_row26_col5, #T_0aae8_row26_col6, #T_0aae8_row26_col7, #T_0aae8_row26_col8, #T_0aae8_row26_col9, #T_0aae8_row26_col10, #T_0aae8_row26_col11, #T_0aae8_row26_col12, #T_0aae8_row26_col13, #T_0aae8_row26_col14, #T_0aae8_row26_col15, #T_0aae8_row26_col16, #T_0aae8_row26_col17, #T_0aae8_row26_col18, #T_0aae8_row26_col19, #T_0aae8_row26_col20, #T_0aae8_row26_col21, #T_0aae8_row26_col22, #T_0aae8_row26_col23, #T_0aae8_row26_col24, #T_0aae8_row26_col25, #T_0aae8_row26_col26, #T_0aae8_row26_col27, #T_0aae8_row27_col0, #T_0aae8_row27_col1, #T_0aae8_row27_col2, #T_0aae8_row27_col3, #T_0aae8_row27_col4, #T_0aae8_row27_col5, #T_0aae8_row27_col6, #T_0aae8_row27_col7, #T_0aae8_row27_col8, #T_0aae8_row27_col9, #T_0aae8_row27_col10, #T_0aae8_row27_col11, #T_0aae8_row27_col12, #T_0aae8_row27_col13, #T_0aae8_row27_col14, #T_0aae8_row27_col15, #T_0aae8_row27_col16, #T_0aae8_row27_col17, #T_0aae8_row27_col18, #T_0aae8_row27_col19, #T_0aae8_row27_col20, #T_0aae8_row27_col21, #T_0aae8_row27_col22, #T_0aae8_row27_col23, #T_0aae8_row27_col24, #T_0aae8_row27_col25, #T_0aae8_row27_col26, #T_0aae8_row27_col27 {
  font-size: 5pt;
  background-color: #ffffff;
  color: #000000;
}
#T_0aae8_row5_col12, #T_0aae8_row11_col14, #T_0aae8_row18_col20, #T_0aae8_row21_col17 {
  font-size: 5pt;
  background-color: #fefefe;
  color: #000000;
}
#T_0aae8_row5_col13, #T_0aae8_row5_col14, #T_0aae8_row5_col15, #T_0aae8_row8_col7, #T_0aae8_row16_col15, #T_0aae8_row22_col6, #T_0aae8_row24_col11 {
  font-size: 5pt;
  background-color: #f7f7f7;
  color: #000000;
}
#T_0aae8_row5_col16 {
  font-size: 5pt;
  background-color: #979797;
  color: #f1f1f1;
}
#T_0aae8_row5_col17 {
  font-size: 5pt;
  background-color: #8c8c8c;
  color: #f1f1f1;
}
#T_0aae8_row5_col18 {
  font-size: 5pt;
  background-color: #616161;
  color: #f1f1f1;
}
#T_0aae8_row5_col19, #T_0aae8_row14_col18 {
  font-size: 5pt;
  background-color: #f3f3f3;
  color: #000000;
}
#T_0aae8_row5_col20 {
  font-size: 5pt;
  background-color: #6b6b6b;
  color: #f1f1f1;
}
#T_0aae8_row5_col21, #T_0aae8_row5_col22, #T_0aae8_row5_col23, #T_0aae8_row6_col13, #T_0aae8_row6_col14, #T_0aae8_row6_col15, #T_0aae8_row6_col16, #T_0aae8_row6_col17, #T_0aae8_row6_col20, #T_0aae8_row7_col9, #T_0aae8_row7_col10, #T_0aae8_row7_col11, #T_0aae8_row7_col12, #T_0aae8_row7_col13, #T_0aae8_row7_col14, #T_0aae8_row7_col15, #T_0aae8_row7_col16, #T_0aae8_row8_col9, #T_0aae8_row8_col10, #T_0aae8_row8_col11, #T_0aae8_row8_col12, #T_0aae8_row8_col13, #T_0aae8_row9_col11, #T_0aae8_row9_col12, #T_0aae8_row10_col12, #T_0aae8_row11_col12, #T_0aae8_row12_col13, #T_0aae8_row14_col15, #T_0aae8_row14_col16, #T_0aae8_row15_col16, #T_0aae8_row15_col17, #T_0aae8_row16_col18, #T_0aae8_row17_col18, #T_0aae8_row17_col19, #T_0aae8_row18_col17, #T_0aae8_row18_col18, #T_0aae8_row19_col15, #T_0aae8_row19_col16, #T_0aae8_row19_col17, #T_0aae8_row20_col13, #T_0aae8_row20_col14, #T_0aae8_row20_col15, #T_0aae8_row20_col16, #T_0aae8_row21_col11, #T_0aae8_row21_col12, #T_0aae8_row21_col13, #T_0aae8_row21_col14, #T_0aae8_row22_col9, #T_0aae8_row22_col10, #T_0aae8_row22_col11, #T_0aae8_row22_col12, #T_0aae8_row23_col7, #T_0aae8_row23_col8, #T_0aae8_row23_col9, #T_0aae8_row23_col10, #T_0aae8_row24_col4, #T_0aae8_row24_col5, #T_0aae8_row24_col6, #T_0aae8_row24_col7 {
  font-size: 5pt;
  background-color: #000000;
  color: #f1f1f1;
}
#T_0aae8_row6_col8 {
  font-size: 5pt;
  background-color: #f1f1f1;
  color: #000000;
}
#T_0aae8_row6_col9 {
  font-size: 5pt;
  background-color: #ededed;
  color: #000000;
}
#T_0aae8_row6_col10, #T_0aae8_row7_col18, #T_0aae8_row16_col16 {
  font-size: 5pt;
  background-color: #bebebe;
  color: #000000;
}
#T_0aae8_row6_col11, #T_0aae8_row9_col17, #T_0aae8_row10_col11 {
  font-size: 5pt;
  background-color: #787878;
  color: #f1f1f1;
}
#T_0aae8_row6_col12 {
  font-size: 5pt;
  background-color: #666666;
  color: #f1f1f1;
}
#T_0aae8_row6_col18, #T_0aae8_row13_col14 {
  font-size: 5pt;
  background-color: #212121;
  color: #f1f1f1;
}
#T_0aae8_row6_col19 {
  font-size: 5pt;
  background-color: #626262;
  color: #f1f1f1;
}
#T_0aae8_row6_col21, #T_0aae8_row14_col14 {
  font-size: 5pt;
  background-color: #0f0f0f;
  color: #f1f1f1;
}
#T_0aae8_row6_col22 {
  font-size: 5pt;
  background-color: #434343;
  color: #f1f1f1;
}
#T_0aae8_row6_col23 {
  font-size: 5pt;
  background-color: #949494;
  color: #f1f1f1;
}
#T_0aae8_row7_col7 {
  font-size: 5pt;
  background-color: #e4e4e4;
  color: #000000;
}
#T_0aae8_row7_col8 {
  font-size: 5pt;
  background-color: #111111;
  color: #f1f1f1;
}
#T_0aae8_row7_col17 {
  font-size: 5pt;
  background-color: #020202;
  color: #f1f1f1;
}
#T_0aae8_row7_col19 {
  font-size: 5pt;
  background-color: #c7c7c7;
  color: #000000;
}
#T_0aae8_row7_col20 {
  font-size: 5pt;
  background-color: #c9c9c9;
  color: #000000;
}
#T_0aae8_row7_col21 {
  font-size: 5pt;
  background-color: #dfdfdf;
  color: #000000;
}
#T_0aae8_row7_col22 {
  font-size: 5pt;
  background-color: #eaeaea;
  color: #000000;
}
#T_0aae8_row8_col8, #T_0aae8_row22_col8 {
  font-size: 5pt;
  background-color: #282828;
  color: #f1f1f1;
}
#T_0aae8_row8_col14, #T_0aae8_row21_col15 {
  font-size: 5pt;
  background-color: #464646;
  color: #f1f1f1;
}
#T_0aae8_row8_col15 {
  font-size: 5pt;
  background-color: #5a5a5a;
  color: #f1f1f1;
}
#T_0aae8_row8_col16 {
  font-size: 5pt;
  background-color: #070707;
  color: #f1f1f1;
}
#T_0aae8_row8_col17, #T_0aae8_row13_col13 {
  font-size: 5pt;
  background-color: #0e0e0e;
  color: #f1f1f1;
}
#T_0aae8_row9_col8, #T_0aae8_row22_col14 {
  font-size: 5pt;
  background-color: #cbcbcb;
  color: #000000;
}
#T_0aae8_row9_col9 {
  font-size: 5pt;
  background-color: #767676;
  color: #f1f1f1;
}
#T_0aae8_row9_col10 {
  font-size: 5pt;
  background-color: #aeaeae;
  color: #000000;
}
#T_0aae8_row9_col13 {
  font-size: 5pt;
  background-color: #3c3c3c;
  color: #f1f1f1;
}
#T_0aae8_row9_col14, #T_0aae8_row12_col11, #T_0aae8_row23_col13 {
  font-size: 5pt;
  background-color: #fafafa;
  color: #000000;
}
#T_0aae8_row9_col16 {
  font-size: 5pt;
  background-color: #e8e8e8;
  color: #000000;
}
#T_0aae8_row10_col9 {
  font-size: 5pt;
  background-color: #f8f8f8;
  color: #000000;
}
#T_0aae8_row10_col13 {
  font-size: 5pt;
  background-color: #c1c1c1;
  color: #000000;
}
#T_0aae8_row11_col11 {
  font-size: 5pt;
  background-color: #888888;
  color: #f1f1f1;
}
#T_0aae8_row11_col13, #T_0aae8_row12_col12, #T_0aae8_row16_col19 {
  font-size: 5pt;
  background-color: #515151;
  color: #f1f1f1;
}
#T_0aae8_row12_col14 {
  font-size: 5pt;
  background-color: #d4d4d4;
  color: #000000;
}
#T_0aae8_row13_col12 {
  font-size: 5pt;
  background-color: #eeeeee;
  color: #000000;
}
#T_0aae8_row13_col15 {
  font-size: 5pt;
  background-color: #717171;
  color: #f1f1f1;
}
#T_0aae8_row13_col16 {
  font-size: 5pt;
  background-color: #adadad;
  color: #000000;
}
#T_0aae8_row14_col13, #T_0aae8_row21_col16 {
  font-size: 5pt;
  background-color: #cacaca;
  color: #000000;
}
#T_0aae8_row14_col17 {
  font-size: 5pt;
  background-color: #9f9f9f;
  color: #f1f1f1;
}
#T_0aae8_row15_col14 {
  font-size: 5pt;
  background-color: #e7e7e7;
  color: #000000;
}
#T_0aae8_row15_col15 {
  font-size: 5pt;
  background-color: #555555;
  color: #f1f1f1;
}
#T_0aae8_row15_col18 {
  font-size: 5pt;
  background-color: #7c7c7c;
  color: #f1f1f1;
}
#T_0aae8_row15_col19 {
  font-size: 5pt;
  background-color: #f2f2f2;
  color: #000000;
}
#T_0aae8_row16_col17 {
  font-size: 5pt;
  background-color: #010101;
  color: #f1f1f1;
}
#T_0aae8_row17_col17 {
  font-size: 5pt;
  background-color: #050505;
  color: #f1f1f1;
}
#T_0aae8_row17_col20 {
  font-size: 5pt;
  background-color: #d9d9d9;
  color: #000000;
}
#T_0aae8_row18_col14 {
  font-size: 5pt;
  background-color: #e6e6e6;
  color: #000000;
}
#T_0aae8_row18_col15 {
  font-size: 5pt;
  background-color: #929292;
  color: #f1f1f1;
}
#T_0aae8_row18_col16 {
  font-size: 5pt;
  background-color: #585858;
  color: #f1f1f1;
}
#T_0aae8_row18_col19 {
  font-size: 5pt;
  background-color: #353535;
  color: #f1f1f1;
}
#T_0aae8_row19_col12 {
  font-size: 5pt;
  background-color: #ebebeb;
  color: #000000;
}
#T_0aae8_row19_col13 {
  font-size: 5pt;
  background-color: #7e7e7e;
  color: #f1f1f1;
}
#T_0aae8_row19_col14 {
  font-size: 5pt;
  background-color: #1c1c1c;
  color: #f1f1f1;
}
#T_0aae8_row19_col18 {
  font-size: 5pt;
  background-color: #030303;
  color: #f1f1f1;
}
#T_0aae8_row19_col19 {
  font-size: 5pt;
  background-color: #565656;
  color: #f1f1f1;
}
#T_0aae8_row20_col10, #T_0aae8_row21_col8 {
  font-size: 5pt;
  background-color: #f4f4f4;
  color: #000000;
}
#T_0aae8_row20_col11 {
  font-size: 5pt;
  background-color: #a5a5a5;
  color: #f1f1f1;
}
#T_0aae8_row20_col12 {
  font-size: 5pt;
  background-color: #252525;
  color: #f1f1f1;
}
#T_0aae8_row20_col17 {
  font-size: 5pt;
  background-color: #414141;
  color: #f1f1f1;
}
#T_0aae8_row20_col18 {
  font-size: 5pt;
  background-color: #cccccc;
  color: #000000;
}
#T_0aae8_row21_col9 {
  font-size: 5pt;
  background-color: #d7d7d7;
  color: #000000;
}
#T_0aae8_row21_col10 {
  font-size: 5pt;
  background-color: #303030;
  color: #f1f1f1;
}
#T_0aae8_row22_col7 {
  font-size: 5pt;
  background-color: #656565;
  color: #f1f1f1;
}
#T_0aae8_row22_col13 {
  font-size: 5pt;
  background-color: #4a4a4a;
  color: #f1f1f1;
}
#T_0aae8_row22_col15 {
  font-size: 5pt;
  background-color: #fbfbfb;
  color: #000000;
}
#T_0aae8_row23_col4 {
  font-size: 5pt;
  background-color: #b4b4b4;
  color: #000000;
}
#T_0aae8_row23_col5 {
  font-size: 5pt;
  background-color: #646464;
  color: #f1f1f1;
}
#T_0aae8_row23_col6 {
  font-size: 5pt;
  background-color: #1f1f1f;
  color: #f1f1f1;
}
#T_0aae8_row23_col11 {
  font-size: 5pt;
  background-color: #0a0a0a;
  color: #f1f1f1;
}
#T_0aae8_row23_col12 {
  font-size: 5pt;
  background-color: #8f8f8f;
  color: #f1f1f1;
}
#T_0aae8_row24_col8 {
  font-size: 5pt;
  background-color: #323232;
  color: #f1f1f1;
}
#T_0aae8_row24_col9 {
  font-size: 5pt;
  background-color: #8d8d8d;
  color: #f1f1f1;
}
#T_0aae8_row24_col10 {
  font-size: 5pt;
  background-color: #909090;
  color: #f1f1f1;
}
</style>
<table id="T_0aae8">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_0aae8_level0_col0" class="col_heading level0 col0" >0</th>
      <th id="T_0aae8_level0_col1" class="col_heading level0 col1" >1</th>
      <th id="T_0aae8_level0_col2" class="col_heading level0 col2" >2</th>
      <th id="T_0aae8_level0_col3" class="col_heading level0 col3" >3</th>
      <th id="T_0aae8_level0_col4" class="col_heading level0 col4" >4</th>
      <th id="T_0aae8_level0_col5" class="col_heading level0 col5" >5</th>
      <th id="T_0aae8_level0_col6" class="col_heading level0 col6" >6</th>
      <th id="T_0aae8_level0_col7" class="col_heading level0 col7" >7</th>
      <th id="T_0aae8_level0_col8" class="col_heading level0 col8" >8</th>
      <th id="T_0aae8_level0_col9" class="col_heading level0 col9" >9</th>
      <th id="T_0aae8_level0_col10" class="col_heading level0 col10" >10</th>
      <th id="T_0aae8_level0_col11" class="col_heading level0 col11" >11</th>
      <th id="T_0aae8_level0_col12" class="col_heading level0 col12" >12</th>
      <th id="T_0aae8_level0_col13" class="col_heading level0 col13" >13</th>
      <th id="T_0aae8_level0_col14" class="col_heading level0 col14" >14</th>
      <th id="T_0aae8_level0_col15" class="col_heading level0 col15" >15</th>
      <th id="T_0aae8_level0_col16" class="col_heading level0 col16" >16</th>
      <th id="T_0aae8_level0_col17" class="col_heading level0 col17" >17</th>
      <th id="T_0aae8_level0_col18" class="col_heading level0 col18" >18</th>
      <th id="T_0aae8_level0_col19" class="col_heading level0 col19" >19</th>
      <th id="T_0aae8_level0_col20" class="col_heading level0 col20" >20</th>
      <th id="T_0aae8_level0_col21" class="col_heading level0 col21" >21</th>
      <th id="T_0aae8_level0_col22" class="col_heading level0 col22" >22</th>
      <th id="T_0aae8_level0_col23" class="col_heading level0 col23" >23</th>
      <th id="T_0aae8_level0_col24" class="col_heading level0 col24" >24</th>
      <th id="T_0aae8_level0_col25" class="col_heading level0 col25" >25</th>
      <th id="T_0aae8_level0_col26" class="col_heading level0 col26" >26</th>
      <th id="T_0aae8_level0_col27" class="col_heading level0 col27" >27</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_0aae8_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_0aae8_row0_col0" class="data row0 col0" >0.00</td>
      <td id="T_0aae8_row0_col1" class="data row0 col1" >0.00</td>
      <td id="T_0aae8_row0_col2" class="data row0 col2" >0.00</td>
      <td id="T_0aae8_row0_col3" class="data row0 col3" >0.00</td>
      <td id="T_0aae8_row0_col4" class="data row0 col4" >0.00</td>
      <td id="T_0aae8_row0_col5" class="data row0 col5" >0.00</td>
      <td id="T_0aae8_row0_col6" class="data row0 col6" >0.00</td>
      <td id="T_0aae8_row0_col7" class="data row0 col7" >0.00</td>
      <td id="T_0aae8_row0_col8" class="data row0 col8" >0.00</td>
      <td id="T_0aae8_row0_col9" class="data row0 col9" >0.00</td>
      <td id="T_0aae8_row0_col10" class="data row0 col10" >0.00</td>
      <td id="T_0aae8_row0_col11" class="data row0 col11" >0.00</td>
      <td id="T_0aae8_row0_col12" class="data row0 col12" >0.00</td>
      <td id="T_0aae8_row0_col13" class="data row0 col13" >0.00</td>
      <td id="T_0aae8_row0_col14" class="data row0 col14" >0.00</td>
      <td id="T_0aae8_row0_col15" class="data row0 col15" >0.00</td>
      <td id="T_0aae8_row0_col16" class="data row0 col16" >0.00</td>
      <td id="T_0aae8_row0_col17" class="data row0 col17" >0.00</td>
      <td id="T_0aae8_row0_col18" class="data row0 col18" >0.00</td>
      <td id="T_0aae8_row0_col19" class="data row0 col19" >0.00</td>
      <td id="T_0aae8_row0_col20" class="data row0 col20" >0.00</td>
      <td id="T_0aae8_row0_col21" class="data row0 col21" >0.00</td>
      <td id="T_0aae8_row0_col22" class="data row0 col22" >0.00</td>
      <td id="T_0aae8_row0_col23" class="data row0 col23" >0.00</td>
      <td id="T_0aae8_row0_col24" class="data row0 col24" >0.00</td>
      <td id="T_0aae8_row0_col25" class="data row0 col25" >0.00</td>
      <td id="T_0aae8_row0_col26" class="data row0 col26" >0.00</td>
      <td id="T_0aae8_row0_col27" class="data row0 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_0aae8_row1_col0" class="data row1 col0" >0.00</td>
      <td id="T_0aae8_row1_col1" class="data row1 col1" >0.00</td>
      <td id="T_0aae8_row1_col2" class="data row1 col2" >0.00</td>
      <td id="T_0aae8_row1_col3" class="data row1 col3" >0.00</td>
      <td id="T_0aae8_row1_col4" class="data row1 col4" >0.00</td>
      <td id="T_0aae8_row1_col5" class="data row1 col5" >0.00</td>
      <td id="T_0aae8_row1_col6" class="data row1 col6" >0.00</td>
      <td id="T_0aae8_row1_col7" class="data row1 col7" >0.00</td>
      <td id="T_0aae8_row1_col8" class="data row1 col8" >0.00</td>
      <td id="T_0aae8_row1_col9" class="data row1 col9" >0.00</td>
      <td id="T_0aae8_row1_col10" class="data row1 col10" >0.00</td>
      <td id="T_0aae8_row1_col11" class="data row1 col11" >0.00</td>
      <td id="T_0aae8_row1_col12" class="data row1 col12" >0.00</td>
      <td id="T_0aae8_row1_col13" class="data row1 col13" >0.00</td>
      <td id="T_0aae8_row1_col14" class="data row1 col14" >0.00</td>
      <td id="T_0aae8_row1_col15" class="data row1 col15" >0.00</td>
      <td id="T_0aae8_row1_col16" class="data row1 col16" >0.00</td>
      <td id="T_0aae8_row1_col17" class="data row1 col17" >0.00</td>
      <td id="T_0aae8_row1_col18" class="data row1 col18" >0.00</td>
      <td id="T_0aae8_row1_col19" class="data row1 col19" >0.00</td>
      <td id="T_0aae8_row1_col20" class="data row1 col20" >0.00</td>
      <td id="T_0aae8_row1_col21" class="data row1 col21" >0.00</td>
      <td id="T_0aae8_row1_col22" class="data row1 col22" >0.00</td>
      <td id="T_0aae8_row1_col23" class="data row1 col23" >0.00</td>
      <td id="T_0aae8_row1_col24" class="data row1 col24" >0.00</td>
      <td id="T_0aae8_row1_col25" class="data row1 col25" >0.00</td>
      <td id="T_0aae8_row1_col26" class="data row1 col26" >0.00</td>
      <td id="T_0aae8_row1_col27" class="data row1 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_0aae8_row2_col0" class="data row2 col0" >0.00</td>
      <td id="T_0aae8_row2_col1" class="data row2 col1" >0.00</td>
      <td id="T_0aae8_row2_col2" class="data row2 col2" >0.00</td>
      <td id="T_0aae8_row2_col3" class="data row2 col3" >0.00</td>
      <td id="T_0aae8_row2_col4" class="data row2 col4" >0.00</td>
      <td id="T_0aae8_row2_col5" class="data row2 col5" >0.00</td>
      <td id="T_0aae8_row2_col6" class="data row2 col6" >0.00</td>
      <td id="T_0aae8_row2_col7" class="data row2 col7" >0.00</td>
      <td id="T_0aae8_row2_col8" class="data row2 col8" >0.00</td>
      <td id="T_0aae8_row2_col9" class="data row2 col9" >0.00</td>
      <td id="T_0aae8_row2_col10" class="data row2 col10" >0.00</td>
      <td id="T_0aae8_row2_col11" class="data row2 col11" >0.00</td>
      <td id="T_0aae8_row2_col12" class="data row2 col12" >0.00</td>
      <td id="T_0aae8_row2_col13" class="data row2 col13" >0.00</td>
      <td id="T_0aae8_row2_col14" class="data row2 col14" >0.00</td>
      <td id="T_0aae8_row2_col15" class="data row2 col15" >0.00</td>
      <td id="T_0aae8_row2_col16" class="data row2 col16" >0.00</td>
      <td id="T_0aae8_row2_col17" class="data row2 col17" >0.00</td>
      <td id="T_0aae8_row2_col18" class="data row2 col18" >0.00</td>
      <td id="T_0aae8_row2_col19" class="data row2 col19" >0.00</td>
      <td id="T_0aae8_row2_col20" class="data row2 col20" >0.00</td>
      <td id="T_0aae8_row2_col21" class="data row2 col21" >0.00</td>
      <td id="T_0aae8_row2_col22" class="data row2 col22" >0.00</td>
      <td id="T_0aae8_row2_col23" class="data row2 col23" >0.00</td>
      <td id="T_0aae8_row2_col24" class="data row2 col24" >0.00</td>
      <td id="T_0aae8_row2_col25" class="data row2 col25" >0.00</td>
      <td id="T_0aae8_row2_col26" class="data row2 col26" >0.00</td>
      <td id="T_0aae8_row2_col27" class="data row2 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_0aae8_row3_col0" class="data row3 col0" >0.00</td>
      <td id="T_0aae8_row3_col1" class="data row3 col1" >0.00</td>
      <td id="T_0aae8_row3_col2" class="data row3 col2" >0.00</td>
      <td id="T_0aae8_row3_col3" class="data row3 col3" >0.00</td>
      <td id="T_0aae8_row3_col4" class="data row3 col4" >0.00</td>
      <td id="T_0aae8_row3_col5" class="data row3 col5" >0.00</td>
      <td id="T_0aae8_row3_col6" class="data row3 col6" >0.00</td>
      <td id="T_0aae8_row3_col7" class="data row3 col7" >0.00</td>
      <td id="T_0aae8_row3_col8" class="data row3 col8" >0.00</td>
      <td id="T_0aae8_row3_col9" class="data row3 col9" >0.00</td>
      <td id="T_0aae8_row3_col10" class="data row3 col10" >0.00</td>
      <td id="T_0aae8_row3_col11" class="data row3 col11" >0.00</td>
      <td id="T_0aae8_row3_col12" class="data row3 col12" >0.00</td>
      <td id="T_0aae8_row3_col13" class="data row3 col13" >0.00</td>
      <td id="T_0aae8_row3_col14" class="data row3 col14" >0.00</td>
      <td id="T_0aae8_row3_col15" class="data row3 col15" >0.00</td>
      <td id="T_0aae8_row3_col16" class="data row3 col16" >0.00</td>
      <td id="T_0aae8_row3_col17" class="data row3 col17" >0.00</td>
      <td id="T_0aae8_row3_col18" class="data row3 col18" >0.00</td>
      <td id="T_0aae8_row3_col19" class="data row3 col19" >0.00</td>
      <td id="T_0aae8_row3_col20" class="data row3 col20" >0.00</td>
      <td id="T_0aae8_row3_col21" class="data row3 col21" >0.00</td>
      <td id="T_0aae8_row3_col22" class="data row3 col22" >0.00</td>
      <td id="T_0aae8_row3_col23" class="data row3 col23" >0.00</td>
      <td id="T_0aae8_row3_col24" class="data row3 col24" >0.00</td>
      <td id="T_0aae8_row3_col25" class="data row3 col25" >0.00</td>
      <td id="T_0aae8_row3_col26" class="data row3 col26" >0.00</td>
      <td id="T_0aae8_row3_col27" class="data row3 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_0aae8_row4_col0" class="data row4 col0" >0.00</td>
      <td id="T_0aae8_row4_col1" class="data row4 col1" >0.00</td>
      <td id="T_0aae8_row4_col2" class="data row4 col2" >0.00</td>
      <td id="T_0aae8_row4_col3" class="data row4 col3" >0.00</td>
      <td id="T_0aae8_row4_col4" class="data row4 col4" >0.00</td>
      <td id="T_0aae8_row4_col5" class="data row4 col5" >0.00</td>
      <td id="T_0aae8_row4_col6" class="data row4 col6" >0.00</td>
      <td id="T_0aae8_row4_col7" class="data row4 col7" >0.00</td>
      <td id="T_0aae8_row4_col8" class="data row4 col8" >0.00</td>
      <td id="T_0aae8_row4_col9" class="data row4 col9" >0.00</td>
      <td id="T_0aae8_row4_col10" class="data row4 col10" >0.00</td>
      <td id="T_0aae8_row4_col11" class="data row4 col11" >0.00</td>
      <td id="T_0aae8_row4_col12" class="data row4 col12" >0.00</td>
      <td id="T_0aae8_row4_col13" class="data row4 col13" >0.00</td>
      <td id="T_0aae8_row4_col14" class="data row4 col14" >0.00</td>
      <td id="T_0aae8_row4_col15" class="data row4 col15" >0.00</td>
      <td id="T_0aae8_row4_col16" class="data row4 col16" >0.00</td>
      <td id="T_0aae8_row4_col17" class="data row4 col17" >0.00</td>
      <td id="T_0aae8_row4_col18" class="data row4 col18" >0.00</td>
      <td id="T_0aae8_row4_col19" class="data row4 col19" >0.00</td>
      <td id="T_0aae8_row4_col20" class="data row4 col20" >0.00</td>
      <td id="T_0aae8_row4_col21" class="data row4 col21" >0.00</td>
      <td id="T_0aae8_row4_col22" class="data row4 col22" >0.00</td>
      <td id="T_0aae8_row4_col23" class="data row4 col23" >0.00</td>
      <td id="T_0aae8_row4_col24" class="data row4 col24" >0.00</td>
      <td id="T_0aae8_row4_col25" class="data row4 col25" >0.00</td>
      <td id="T_0aae8_row4_col26" class="data row4 col26" >0.00</td>
      <td id="T_0aae8_row4_col27" class="data row4 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_0aae8_row5_col0" class="data row5 col0" >0.00</td>
      <td id="T_0aae8_row5_col1" class="data row5 col1" >0.00</td>
      <td id="T_0aae8_row5_col2" class="data row5 col2" >0.00</td>
      <td id="T_0aae8_row5_col3" class="data row5 col3" >0.00</td>
      <td id="T_0aae8_row5_col4" class="data row5 col4" >0.00</td>
      <td id="T_0aae8_row5_col5" class="data row5 col5" >0.00</td>
      <td id="T_0aae8_row5_col6" class="data row5 col6" >0.00</td>
      <td id="T_0aae8_row5_col7" class="data row5 col7" >0.00</td>
      <td id="T_0aae8_row5_col8" class="data row5 col8" >0.00</td>
      <td id="T_0aae8_row5_col9" class="data row5 col9" >0.00</td>
      <td id="T_0aae8_row5_col10" class="data row5 col10" >0.00</td>
      <td id="T_0aae8_row5_col11" class="data row5 col11" >0.00</td>
      <td id="T_0aae8_row5_col12" class="data row5 col12" >0.01</td>
      <td id="T_0aae8_row5_col13" class="data row5 col13" >0.07</td>
      <td id="T_0aae8_row5_col14" class="data row5 col14" >0.07</td>
      <td id="T_0aae8_row5_col15" class="data row5 col15" >0.07</td>
      <td id="T_0aae8_row5_col16" class="data row5 col16" >0.49</td>
      <td id="T_0aae8_row5_col17" class="data row5 col17" >0.53</td>
      <td id="T_0aae8_row5_col18" class="data row5 col18" >0.69</td>
      <td id="T_0aae8_row5_col19" class="data row5 col19" >0.10</td>
      <td id="T_0aae8_row5_col20" class="data row5 col20" >0.65</td>
      <td id="T_0aae8_row5_col21" class="data row5 col21" >1.00</td>
      <td id="T_0aae8_row5_col22" class="data row5 col22" >0.97</td>
      <td id="T_0aae8_row5_col23" class="data row5 col23" >0.50</td>
      <td id="T_0aae8_row5_col24" class="data row5 col24" >0.00</td>
      <td id="T_0aae8_row5_col25" class="data row5 col25" >0.00</td>
      <td id="T_0aae8_row5_col26" class="data row5 col26" >0.00</td>
      <td id="T_0aae8_row5_col27" class="data row5 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_0aae8_row6_col0" class="data row6 col0" >0.00</td>
      <td id="T_0aae8_row6_col1" class="data row6 col1" >0.00</td>
      <td id="T_0aae8_row6_col2" class="data row6 col2" >0.00</td>
      <td id="T_0aae8_row6_col3" class="data row6 col3" >0.00</td>
      <td id="T_0aae8_row6_col4" class="data row6 col4" >0.00</td>
      <td id="T_0aae8_row6_col5" class="data row6 col5" >0.00</td>
      <td id="T_0aae8_row6_col6" class="data row6 col6" >0.00</td>
      <td id="T_0aae8_row6_col7" class="data row6 col7" >0.00</td>
      <td id="T_0aae8_row6_col8" class="data row6 col8" >0.12</td>
      <td id="T_0aae8_row6_col9" class="data row6 col9" >0.14</td>
      <td id="T_0aae8_row6_col10" class="data row6 col10" >0.37</td>
      <td id="T_0aae8_row6_col11" class="data row6 col11" >0.60</td>
      <td id="T_0aae8_row6_col12" class="data row6 col12" >0.67</td>
      <td id="T_0aae8_row6_col13" class="data row6 col13" >0.99</td>
      <td id="T_0aae8_row6_col14" class="data row6 col14" >0.99</td>
      <td id="T_0aae8_row6_col15" class="data row6 col15" >0.99</td>
      <td id="T_0aae8_row6_col16" class="data row6 col16" >0.99</td>
      <td id="T_0aae8_row6_col17" class="data row6 col17" >0.99</td>
      <td id="T_0aae8_row6_col18" class="data row6 col18" >0.88</td>
      <td id="T_0aae8_row6_col19" class="data row6 col19" >0.67</td>
      <td id="T_0aae8_row6_col20" class="data row6 col20" >0.99</td>
      <td id="T_0aae8_row6_col21" class="data row6 col21" >0.95</td>
      <td id="T_0aae8_row6_col22" class="data row6 col22" >0.76</td>
      <td id="T_0aae8_row6_col23" class="data row6 col23" >0.25</td>
      <td id="T_0aae8_row6_col24" class="data row6 col24" >0.00</td>
      <td id="T_0aae8_row6_col25" class="data row6 col25" >0.00</td>
      <td id="T_0aae8_row6_col26" class="data row6 col26" >0.00</td>
      <td id="T_0aae8_row6_col27" class="data row6 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_0aae8_row7_col0" class="data row7 col0" >0.00</td>
      <td id="T_0aae8_row7_col1" class="data row7 col1" >0.00</td>
      <td id="T_0aae8_row7_col2" class="data row7 col2" >0.00</td>
      <td id="T_0aae8_row7_col3" class="data row7 col3" >0.00</td>
      <td id="T_0aae8_row7_col4" class="data row7 col4" >0.00</td>
      <td id="T_0aae8_row7_col5" class="data row7 col5" >0.00</td>
      <td id="T_0aae8_row7_col6" class="data row7 col6" >0.00</td>
      <td id="T_0aae8_row7_col7" class="data row7 col7" >0.19</td>
      <td id="T_0aae8_row7_col8" class="data row7 col8" >0.93</td>
      <td id="T_0aae8_row7_col9" class="data row7 col9" >0.99</td>
      <td id="T_0aae8_row7_col10" class="data row7 col10" >0.99</td>
      <td id="T_0aae8_row7_col11" class="data row7 col11" >0.99</td>
      <td id="T_0aae8_row7_col12" class="data row7 col12" >0.99</td>
      <td id="T_0aae8_row7_col13" class="data row7 col13" >0.99</td>
      <td id="T_0aae8_row7_col14" class="data row7 col14" >0.99</td>
      <td id="T_0aae8_row7_col15" class="data row7 col15" >0.99</td>
      <td id="T_0aae8_row7_col16" class="data row7 col16" >0.99</td>
      <td id="T_0aae8_row7_col17" class="data row7 col17" >0.98</td>
      <td id="T_0aae8_row7_col18" class="data row7 col18" >0.36</td>
      <td id="T_0aae8_row7_col19" class="data row7 col19" >0.32</td>
      <td id="T_0aae8_row7_col20" class="data row7 col20" >0.32</td>
      <td id="T_0aae8_row7_col21" class="data row7 col21" >0.22</td>
      <td id="T_0aae8_row7_col22" class="data row7 col22" >0.15</td>
      <td id="T_0aae8_row7_col23" class="data row7 col23" >0.00</td>
      <td id="T_0aae8_row7_col24" class="data row7 col24" >0.00</td>
      <td id="T_0aae8_row7_col25" class="data row7 col25" >0.00</td>
      <td id="T_0aae8_row7_col26" class="data row7 col26" >0.00</td>
      <td id="T_0aae8_row7_col27" class="data row7 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_0aae8_row8_col0" class="data row8 col0" >0.00</td>
      <td id="T_0aae8_row8_col1" class="data row8 col1" >0.00</td>
      <td id="T_0aae8_row8_col2" class="data row8 col2" >0.00</td>
      <td id="T_0aae8_row8_col3" class="data row8 col3" >0.00</td>
      <td id="T_0aae8_row8_col4" class="data row8 col4" >0.00</td>
      <td id="T_0aae8_row8_col5" class="data row8 col5" >0.00</td>
      <td id="T_0aae8_row8_col6" class="data row8 col6" >0.00</td>
      <td id="T_0aae8_row8_col7" class="data row8 col7" >0.07</td>
      <td id="T_0aae8_row8_col8" class="data row8 col8" >0.86</td>
      <td id="T_0aae8_row8_col9" class="data row8 col9" >0.99</td>
      <td id="T_0aae8_row8_col10" class="data row8 col10" >0.99</td>
      <td id="T_0aae8_row8_col11" class="data row8 col11" >0.99</td>
      <td id="T_0aae8_row8_col12" class="data row8 col12" >0.99</td>
      <td id="T_0aae8_row8_col13" class="data row8 col13" >0.99</td>
      <td id="T_0aae8_row8_col14" class="data row8 col14" >0.78</td>
      <td id="T_0aae8_row8_col15" class="data row8 col15" >0.71</td>
      <td id="T_0aae8_row8_col16" class="data row8 col16" >0.97</td>
      <td id="T_0aae8_row8_col17" class="data row8 col17" >0.95</td>
      <td id="T_0aae8_row8_col18" class="data row8 col18" >0.00</td>
      <td id="T_0aae8_row8_col19" class="data row8 col19" >0.00</td>
      <td id="T_0aae8_row8_col20" class="data row8 col20" >0.00</td>
      <td id="T_0aae8_row8_col21" class="data row8 col21" >0.00</td>
      <td id="T_0aae8_row8_col22" class="data row8 col22" >0.00</td>
      <td id="T_0aae8_row8_col23" class="data row8 col23" >0.00</td>
      <td id="T_0aae8_row8_col24" class="data row8 col24" >0.00</td>
      <td id="T_0aae8_row8_col25" class="data row8 col25" >0.00</td>
      <td id="T_0aae8_row8_col26" class="data row8 col26" >0.00</td>
      <td id="T_0aae8_row8_col27" class="data row8 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_0aae8_row9_col0" class="data row9 col0" >0.00</td>
      <td id="T_0aae8_row9_col1" class="data row9 col1" >0.00</td>
      <td id="T_0aae8_row9_col2" class="data row9 col2" >0.00</td>
      <td id="T_0aae8_row9_col3" class="data row9 col3" >0.00</td>
      <td id="T_0aae8_row9_col4" class="data row9 col4" >0.00</td>
      <td id="T_0aae8_row9_col5" class="data row9 col5" >0.00</td>
      <td id="T_0aae8_row9_col6" class="data row9 col6" >0.00</td>
      <td id="T_0aae8_row9_col7" class="data row9 col7" >0.00</td>
      <td id="T_0aae8_row9_col8" class="data row9 col8" >0.31</td>
      <td id="T_0aae8_row9_col9" class="data row9 col9" >0.61</td>
      <td id="T_0aae8_row9_col10" class="data row9 col10" >0.42</td>
      <td id="T_0aae8_row9_col11" class="data row9 col11" >0.99</td>
      <td id="T_0aae8_row9_col12" class="data row9 col12" >0.99</td>
      <td id="T_0aae8_row9_col13" class="data row9 col13" >0.80</td>
      <td id="T_0aae8_row9_col14" class="data row9 col14" >0.04</td>
      <td id="T_0aae8_row9_col15" class="data row9 col15" >0.00</td>
      <td id="T_0aae8_row9_col16" class="data row9 col16" >0.17</td>
      <td id="T_0aae8_row9_col17" class="data row9 col17" >0.60</td>
      <td id="T_0aae8_row9_col18" class="data row9 col18" >0.00</td>
      <td id="T_0aae8_row9_col19" class="data row9 col19" >0.00</td>
      <td id="T_0aae8_row9_col20" class="data row9 col20" >0.00</td>
      <td id="T_0aae8_row9_col21" class="data row9 col21" >0.00</td>
      <td id="T_0aae8_row9_col22" class="data row9 col22" >0.00</td>
      <td id="T_0aae8_row9_col23" class="data row9 col23" >0.00</td>
      <td id="T_0aae8_row9_col24" class="data row9 col24" >0.00</td>
      <td id="T_0aae8_row9_col25" class="data row9 col25" >0.00</td>
      <td id="T_0aae8_row9_col26" class="data row9 col26" >0.00</td>
      <td id="T_0aae8_row9_col27" class="data row9 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_0aae8_row10_col0" class="data row10 col0" >0.00</td>
      <td id="T_0aae8_row10_col1" class="data row10 col1" >0.00</td>
      <td id="T_0aae8_row10_col2" class="data row10 col2" >0.00</td>
      <td id="T_0aae8_row10_col3" class="data row10 col3" >0.00</td>
      <td id="T_0aae8_row10_col4" class="data row10 col4" >0.00</td>
      <td id="T_0aae8_row10_col5" class="data row10 col5" >0.00</td>
      <td id="T_0aae8_row10_col6" class="data row10 col6" >0.00</td>
      <td id="T_0aae8_row10_col7" class="data row10 col7" >0.00</td>
      <td id="T_0aae8_row10_col8" class="data row10 col8" >0.00</td>
      <td id="T_0aae8_row10_col9" class="data row10 col9" >0.05</td>
      <td id="T_0aae8_row10_col10" class="data row10 col10" >0.00</td>
      <td id="T_0aae8_row10_col11" class="data row10 col11" >0.60</td>
      <td id="T_0aae8_row10_col12" class="data row10 col12" >0.99</td>
      <td id="T_0aae8_row10_col13" class="data row10 col13" >0.35</td>
      <td id="T_0aae8_row10_col14" class="data row10 col14" >0.00</td>
      <td id="T_0aae8_row10_col15" class="data row10 col15" >0.00</td>
      <td id="T_0aae8_row10_col16" class="data row10 col16" >0.00</td>
      <td id="T_0aae8_row10_col17" class="data row10 col17" >0.00</td>
      <td id="T_0aae8_row10_col18" class="data row10 col18" >0.00</td>
      <td id="T_0aae8_row10_col19" class="data row10 col19" >0.00</td>
      <td id="T_0aae8_row10_col20" class="data row10 col20" >0.00</td>
      <td id="T_0aae8_row10_col21" class="data row10 col21" >0.00</td>
      <td id="T_0aae8_row10_col22" class="data row10 col22" >0.00</td>
      <td id="T_0aae8_row10_col23" class="data row10 col23" >0.00</td>
      <td id="T_0aae8_row10_col24" class="data row10 col24" >0.00</td>
      <td id="T_0aae8_row10_col25" class="data row10 col25" >0.00</td>
      <td id="T_0aae8_row10_col26" class="data row10 col26" >0.00</td>
      <td id="T_0aae8_row10_col27" class="data row10 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_0aae8_row11_col0" class="data row11 col0" >0.00</td>
      <td id="T_0aae8_row11_col1" class="data row11 col1" >0.00</td>
      <td id="T_0aae8_row11_col2" class="data row11 col2" >0.00</td>
      <td id="T_0aae8_row11_col3" class="data row11 col3" >0.00</td>
      <td id="T_0aae8_row11_col4" class="data row11 col4" >0.00</td>
      <td id="T_0aae8_row11_col5" class="data row11 col5" >0.00</td>
      <td id="T_0aae8_row11_col6" class="data row11 col6" >0.00</td>
      <td id="T_0aae8_row11_col7" class="data row11 col7" >0.00</td>
      <td id="T_0aae8_row11_col8" class="data row11 col8" >0.00</td>
      <td id="T_0aae8_row11_col9" class="data row11 col9" >0.00</td>
      <td id="T_0aae8_row11_col10" class="data row11 col10" >0.00</td>
      <td id="T_0aae8_row11_col11" class="data row11 col11" >0.55</td>
      <td id="T_0aae8_row11_col12" class="data row11 col12" >0.99</td>
      <td id="T_0aae8_row11_col13" class="data row11 col13" >0.75</td>
      <td id="T_0aae8_row11_col14" class="data row11 col14" >0.01</td>
      <td id="T_0aae8_row11_col15" class="data row11 col15" >0.00</td>
      <td id="T_0aae8_row11_col16" class="data row11 col16" >0.00</td>
      <td id="T_0aae8_row11_col17" class="data row11 col17" >0.00</td>
      <td id="T_0aae8_row11_col18" class="data row11 col18" >0.00</td>
      <td id="T_0aae8_row11_col19" class="data row11 col19" >0.00</td>
      <td id="T_0aae8_row11_col20" class="data row11 col20" >0.00</td>
      <td id="T_0aae8_row11_col21" class="data row11 col21" >0.00</td>
      <td id="T_0aae8_row11_col22" class="data row11 col22" >0.00</td>
      <td id="T_0aae8_row11_col23" class="data row11 col23" >0.00</td>
      <td id="T_0aae8_row11_col24" class="data row11 col24" >0.00</td>
      <td id="T_0aae8_row11_col25" class="data row11 col25" >0.00</td>
      <td id="T_0aae8_row11_col26" class="data row11 col26" >0.00</td>
      <td id="T_0aae8_row11_col27" class="data row11 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_0aae8_row12_col0" class="data row12 col0" >0.00</td>
      <td id="T_0aae8_row12_col1" class="data row12 col1" >0.00</td>
      <td id="T_0aae8_row12_col2" class="data row12 col2" >0.00</td>
      <td id="T_0aae8_row12_col3" class="data row12 col3" >0.00</td>
      <td id="T_0aae8_row12_col4" class="data row12 col4" >0.00</td>
      <td id="T_0aae8_row12_col5" class="data row12 col5" >0.00</td>
      <td id="T_0aae8_row12_col6" class="data row12 col6" >0.00</td>
      <td id="T_0aae8_row12_col7" class="data row12 col7" >0.00</td>
      <td id="T_0aae8_row12_col8" class="data row12 col8" >0.00</td>
      <td id="T_0aae8_row12_col9" class="data row12 col9" >0.00</td>
      <td id="T_0aae8_row12_col10" class="data row12 col10" >0.00</td>
      <td id="T_0aae8_row12_col11" class="data row12 col11" >0.04</td>
      <td id="T_0aae8_row12_col12" class="data row12 col12" >0.75</td>
      <td id="T_0aae8_row12_col13" class="data row12 col13" >0.99</td>
      <td id="T_0aae8_row12_col14" class="data row12 col14" >0.27</td>
      <td id="T_0aae8_row12_col15" class="data row12 col15" >0.00</td>
      <td id="T_0aae8_row12_col16" class="data row12 col16" >0.00</td>
      <td id="T_0aae8_row12_col17" class="data row12 col17" >0.00</td>
      <td id="T_0aae8_row12_col18" class="data row12 col18" >0.00</td>
      <td id="T_0aae8_row12_col19" class="data row12 col19" >0.00</td>
      <td id="T_0aae8_row12_col20" class="data row12 col20" >0.00</td>
      <td id="T_0aae8_row12_col21" class="data row12 col21" >0.00</td>
      <td id="T_0aae8_row12_col22" class="data row12 col22" >0.00</td>
      <td id="T_0aae8_row12_col23" class="data row12 col23" >0.00</td>
      <td id="T_0aae8_row12_col24" class="data row12 col24" >0.00</td>
      <td id="T_0aae8_row12_col25" class="data row12 col25" >0.00</td>
      <td id="T_0aae8_row12_col26" class="data row12 col26" >0.00</td>
      <td id="T_0aae8_row12_col27" class="data row12 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_0aae8_row13_col0" class="data row13 col0" >0.00</td>
      <td id="T_0aae8_row13_col1" class="data row13 col1" >0.00</td>
      <td id="T_0aae8_row13_col2" class="data row13 col2" >0.00</td>
      <td id="T_0aae8_row13_col3" class="data row13 col3" >0.00</td>
      <td id="T_0aae8_row13_col4" class="data row13 col4" >0.00</td>
      <td id="T_0aae8_row13_col5" class="data row13 col5" >0.00</td>
      <td id="T_0aae8_row13_col6" class="data row13 col6" >0.00</td>
      <td id="T_0aae8_row13_col7" class="data row13 col7" >0.00</td>
      <td id="T_0aae8_row13_col8" class="data row13 col8" >0.00</td>
      <td id="T_0aae8_row13_col9" class="data row13 col9" >0.00</td>
      <td id="T_0aae8_row13_col10" class="data row13 col10" >0.00</td>
      <td id="T_0aae8_row13_col11" class="data row13 col11" >0.00</td>
      <td id="T_0aae8_row13_col12" class="data row13 col12" >0.14</td>
      <td id="T_0aae8_row13_col13" class="data row13 col13" >0.95</td>
      <td id="T_0aae8_row13_col14" class="data row13 col14" >0.88</td>
      <td id="T_0aae8_row13_col15" class="data row13 col15" >0.63</td>
      <td id="T_0aae8_row13_col16" class="data row13 col16" >0.42</td>
      <td id="T_0aae8_row13_col17" class="data row13 col17" >0.00</td>
      <td id="T_0aae8_row13_col18" class="data row13 col18" >0.00</td>
      <td id="T_0aae8_row13_col19" class="data row13 col19" >0.00</td>
      <td id="T_0aae8_row13_col20" class="data row13 col20" >0.00</td>
      <td id="T_0aae8_row13_col21" class="data row13 col21" >0.00</td>
      <td id="T_0aae8_row13_col22" class="data row13 col22" >0.00</td>
      <td id="T_0aae8_row13_col23" class="data row13 col23" >0.00</td>
      <td id="T_0aae8_row13_col24" class="data row13 col24" >0.00</td>
      <td id="T_0aae8_row13_col25" class="data row13 col25" >0.00</td>
      <td id="T_0aae8_row13_col26" class="data row13 col26" >0.00</td>
      <td id="T_0aae8_row13_col27" class="data row13 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_0aae8_row14_col0" class="data row14 col0" >0.00</td>
      <td id="T_0aae8_row14_col1" class="data row14 col1" >0.00</td>
      <td id="T_0aae8_row14_col2" class="data row14 col2" >0.00</td>
      <td id="T_0aae8_row14_col3" class="data row14 col3" >0.00</td>
      <td id="T_0aae8_row14_col4" class="data row14 col4" >0.00</td>
      <td id="T_0aae8_row14_col5" class="data row14 col5" >0.00</td>
      <td id="T_0aae8_row14_col6" class="data row14 col6" >0.00</td>
      <td id="T_0aae8_row14_col7" class="data row14 col7" >0.00</td>
      <td id="T_0aae8_row14_col8" class="data row14 col8" >0.00</td>
      <td id="T_0aae8_row14_col9" class="data row14 col9" >0.00</td>
      <td id="T_0aae8_row14_col10" class="data row14 col10" >0.00</td>
      <td id="T_0aae8_row14_col11" class="data row14 col11" >0.00</td>
      <td id="T_0aae8_row14_col12" class="data row14 col12" >0.00</td>
      <td id="T_0aae8_row14_col13" class="data row14 col13" >0.32</td>
      <td id="T_0aae8_row14_col14" class="data row14 col14" >0.94</td>
      <td id="T_0aae8_row14_col15" class="data row14 col15" >0.99</td>
      <td id="T_0aae8_row14_col16" class="data row14 col16" >0.99</td>
      <td id="T_0aae8_row14_col17" class="data row14 col17" >0.47</td>
      <td id="T_0aae8_row14_col18" class="data row14 col18" >0.10</td>
      <td id="T_0aae8_row14_col19" class="data row14 col19" >0.00</td>
      <td id="T_0aae8_row14_col20" class="data row14 col20" >0.00</td>
      <td id="T_0aae8_row14_col21" class="data row14 col21" >0.00</td>
      <td id="T_0aae8_row14_col22" class="data row14 col22" >0.00</td>
      <td id="T_0aae8_row14_col23" class="data row14 col23" >0.00</td>
      <td id="T_0aae8_row14_col24" class="data row14 col24" >0.00</td>
      <td id="T_0aae8_row14_col25" class="data row14 col25" >0.00</td>
      <td id="T_0aae8_row14_col26" class="data row14 col26" >0.00</td>
      <td id="T_0aae8_row14_col27" class="data row14 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_0aae8_row15_col0" class="data row15 col0" >0.00</td>
      <td id="T_0aae8_row15_col1" class="data row15 col1" >0.00</td>
      <td id="T_0aae8_row15_col2" class="data row15 col2" >0.00</td>
      <td id="T_0aae8_row15_col3" class="data row15 col3" >0.00</td>
      <td id="T_0aae8_row15_col4" class="data row15 col4" >0.00</td>
      <td id="T_0aae8_row15_col5" class="data row15 col5" >0.00</td>
      <td id="T_0aae8_row15_col6" class="data row15 col6" >0.00</td>
      <td id="T_0aae8_row15_col7" class="data row15 col7" >0.00</td>
      <td id="T_0aae8_row15_col8" class="data row15 col8" >0.00</td>
      <td id="T_0aae8_row15_col9" class="data row15 col9" >0.00</td>
      <td id="T_0aae8_row15_col10" class="data row15 col10" >0.00</td>
      <td id="T_0aae8_row15_col11" class="data row15 col11" >0.00</td>
      <td id="T_0aae8_row15_col12" class="data row15 col12" >0.00</td>
      <td id="T_0aae8_row15_col13" class="data row15 col13" >0.00</td>
      <td id="T_0aae8_row15_col14" class="data row15 col14" >0.18</td>
      <td id="T_0aae8_row15_col15" class="data row15 col15" >0.73</td>
      <td id="T_0aae8_row15_col16" class="data row15 col16" >0.99</td>
      <td id="T_0aae8_row15_col17" class="data row15 col17" >0.99</td>
      <td id="T_0aae8_row15_col18" class="data row15 col18" >0.59</td>
      <td id="T_0aae8_row15_col19" class="data row15 col19" >0.11</td>
      <td id="T_0aae8_row15_col20" class="data row15 col20" >0.00</td>
      <td id="T_0aae8_row15_col21" class="data row15 col21" >0.00</td>
      <td id="T_0aae8_row15_col22" class="data row15 col22" >0.00</td>
      <td id="T_0aae8_row15_col23" class="data row15 col23" >0.00</td>
      <td id="T_0aae8_row15_col24" class="data row15 col24" >0.00</td>
      <td id="T_0aae8_row15_col25" class="data row15 col25" >0.00</td>
      <td id="T_0aae8_row15_col26" class="data row15 col26" >0.00</td>
      <td id="T_0aae8_row15_col27" class="data row15 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_0aae8_row16_col0" class="data row16 col0" >0.00</td>
      <td id="T_0aae8_row16_col1" class="data row16 col1" >0.00</td>
      <td id="T_0aae8_row16_col2" class="data row16 col2" >0.00</td>
      <td id="T_0aae8_row16_col3" class="data row16 col3" >0.00</td>
      <td id="T_0aae8_row16_col4" class="data row16 col4" >0.00</td>
      <td id="T_0aae8_row16_col5" class="data row16 col5" >0.00</td>
      <td id="T_0aae8_row16_col6" class="data row16 col6" >0.00</td>
      <td id="T_0aae8_row16_col7" class="data row16 col7" >0.00</td>
      <td id="T_0aae8_row16_col8" class="data row16 col8" >0.00</td>
      <td id="T_0aae8_row16_col9" class="data row16 col9" >0.00</td>
      <td id="T_0aae8_row16_col10" class="data row16 col10" >0.00</td>
      <td id="T_0aae8_row16_col11" class="data row16 col11" >0.00</td>
      <td id="T_0aae8_row16_col12" class="data row16 col12" >0.00</td>
      <td id="T_0aae8_row16_col13" class="data row16 col13" >0.00</td>
      <td id="T_0aae8_row16_col14" class="data row16 col14" >0.00</td>
      <td id="T_0aae8_row16_col15" class="data row16 col15" >0.06</td>
      <td id="T_0aae8_row16_col16" class="data row16 col16" >0.36</td>
      <td id="T_0aae8_row16_col17" class="data row16 col17" >0.99</td>
      <td id="T_0aae8_row16_col18" class="data row16 col18" >0.99</td>
      <td id="T_0aae8_row16_col19" class="data row16 col19" >0.73</td>
      <td id="T_0aae8_row16_col20" class="data row16 col20" >0.00</td>
      <td id="T_0aae8_row16_col21" class="data row16 col21" >0.00</td>
      <td id="T_0aae8_row16_col22" class="data row16 col22" >0.00</td>
      <td id="T_0aae8_row16_col23" class="data row16 col23" >0.00</td>
      <td id="T_0aae8_row16_col24" class="data row16 col24" >0.00</td>
      <td id="T_0aae8_row16_col25" class="data row16 col25" >0.00</td>
      <td id="T_0aae8_row16_col26" class="data row16 col26" >0.00</td>
      <td id="T_0aae8_row16_col27" class="data row16 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_0aae8_row17_col0" class="data row17 col0" >0.00</td>
      <td id="T_0aae8_row17_col1" class="data row17 col1" >0.00</td>
      <td id="T_0aae8_row17_col2" class="data row17 col2" >0.00</td>
      <td id="T_0aae8_row17_col3" class="data row17 col3" >0.00</td>
      <td id="T_0aae8_row17_col4" class="data row17 col4" >0.00</td>
      <td id="T_0aae8_row17_col5" class="data row17 col5" >0.00</td>
      <td id="T_0aae8_row17_col6" class="data row17 col6" >0.00</td>
      <td id="T_0aae8_row17_col7" class="data row17 col7" >0.00</td>
      <td id="T_0aae8_row17_col8" class="data row17 col8" >0.00</td>
      <td id="T_0aae8_row17_col9" class="data row17 col9" >0.00</td>
      <td id="T_0aae8_row17_col10" class="data row17 col10" >0.00</td>
      <td id="T_0aae8_row17_col11" class="data row17 col11" >0.00</td>
      <td id="T_0aae8_row17_col12" class="data row17 col12" >0.00</td>
      <td id="T_0aae8_row17_col13" class="data row17 col13" >0.00</td>
      <td id="T_0aae8_row17_col14" class="data row17 col14" >0.00</td>
      <td id="T_0aae8_row17_col15" class="data row17 col15" >0.00</td>
      <td id="T_0aae8_row17_col16" class="data row17 col16" >0.00</td>
      <td id="T_0aae8_row17_col17" class="data row17 col17" >0.98</td>
      <td id="T_0aae8_row17_col18" class="data row17 col18" >0.99</td>
      <td id="T_0aae8_row17_col19" class="data row17 col19" >0.98</td>
      <td id="T_0aae8_row17_col20" class="data row17 col20" >0.25</td>
      <td id="T_0aae8_row17_col21" class="data row17 col21" >0.00</td>
      <td id="T_0aae8_row17_col22" class="data row17 col22" >0.00</td>
      <td id="T_0aae8_row17_col23" class="data row17 col23" >0.00</td>
      <td id="T_0aae8_row17_col24" class="data row17 col24" >0.00</td>
      <td id="T_0aae8_row17_col25" class="data row17 col25" >0.00</td>
      <td id="T_0aae8_row17_col26" class="data row17 col26" >0.00</td>
      <td id="T_0aae8_row17_col27" class="data row17 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_0aae8_row18_col0" class="data row18 col0" >0.00</td>
      <td id="T_0aae8_row18_col1" class="data row18 col1" >0.00</td>
      <td id="T_0aae8_row18_col2" class="data row18 col2" >0.00</td>
      <td id="T_0aae8_row18_col3" class="data row18 col3" >0.00</td>
      <td id="T_0aae8_row18_col4" class="data row18 col4" >0.00</td>
      <td id="T_0aae8_row18_col5" class="data row18 col5" >0.00</td>
      <td id="T_0aae8_row18_col6" class="data row18 col6" >0.00</td>
      <td id="T_0aae8_row18_col7" class="data row18 col7" >0.00</td>
      <td id="T_0aae8_row18_col8" class="data row18 col8" >0.00</td>
      <td id="T_0aae8_row18_col9" class="data row18 col9" >0.00</td>
      <td id="T_0aae8_row18_col10" class="data row18 col10" >0.00</td>
      <td id="T_0aae8_row18_col11" class="data row18 col11" >0.00</td>
      <td id="T_0aae8_row18_col12" class="data row18 col12" >0.00</td>
      <td id="T_0aae8_row18_col13" class="data row18 col13" >0.00</td>
      <td id="T_0aae8_row18_col14" class="data row18 col14" >0.18</td>
      <td id="T_0aae8_row18_col15" class="data row18 col15" >0.51</td>
      <td id="T_0aae8_row18_col16" class="data row18 col16" >0.72</td>
      <td id="T_0aae8_row18_col17" class="data row18 col17" >0.99</td>
      <td id="T_0aae8_row18_col18" class="data row18 col18" >0.99</td>
      <td id="T_0aae8_row18_col19" class="data row18 col19" >0.81</td>
      <td id="T_0aae8_row18_col20" class="data row18 col20" >0.01</td>
      <td id="T_0aae8_row18_col21" class="data row18 col21" >0.00</td>
      <td id="T_0aae8_row18_col22" class="data row18 col22" >0.00</td>
      <td id="T_0aae8_row18_col23" class="data row18 col23" >0.00</td>
      <td id="T_0aae8_row18_col24" class="data row18 col24" >0.00</td>
      <td id="T_0aae8_row18_col25" class="data row18 col25" >0.00</td>
      <td id="T_0aae8_row18_col26" class="data row18 col26" >0.00</td>
      <td id="T_0aae8_row18_col27" class="data row18 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_0aae8_row19_col0" class="data row19 col0" >0.00</td>
      <td id="T_0aae8_row19_col1" class="data row19 col1" >0.00</td>
      <td id="T_0aae8_row19_col2" class="data row19 col2" >0.00</td>
      <td id="T_0aae8_row19_col3" class="data row19 col3" >0.00</td>
      <td id="T_0aae8_row19_col4" class="data row19 col4" >0.00</td>
      <td id="T_0aae8_row19_col5" class="data row19 col5" >0.00</td>
      <td id="T_0aae8_row19_col6" class="data row19 col6" >0.00</td>
      <td id="T_0aae8_row19_col7" class="data row19 col7" >0.00</td>
      <td id="T_0aae8_row19_col8" class="data row19 col8" >0.00</td>
      <td id="T_0aae8_row19_col9" class="data row19 col9" >0.00</td>
      <td id="T_0aae8_row19_col10" class="data row19 col10" >0.00</td>
      <td id="T_0aae8_row19_col11" class="data row19 col11" >0.00</td>
      <td id="T_0aae8_row19_col12" class="data row19 col12" >0.15</td>
      <td id="T_0aae8_row19_col13" class="data row19 col13" >0.58</td>
      <td id="T_0aae8_row19_col14" class="data row19 col14" >0.90</td>
      <td id="T_0aae8_row19_col15" class="data row19 col15" >0.99</td>
      <td id="T_0aae8_row19_col16" class="data row19 col16" >0.99</td>
      <td id="T_0aae8_row19_col17" class="data row19 col17" >0.99</td>
      <td id="T_0aae8_row19_col18" class="data row19 col18" >0.98</td>
      <td id="T_0aae8_row19_col19" class="data row19 col19" >0.71</td>
      <td id="T_0aae8_row19_col20" class="data row19 col20" >0.00</td>
      <td id="T_0aae8_row19_col21" class="data row19 col21" >0.00</td>
      <td id="T_0aae8_row19_col22" class="data row19 col22" >0.00</td>
      <td id="T_0aae8_row19_col23" class="data row19 col23" >0.00</td>
      <td id="T_0aae8_row19_col24" class="data row19 col24" >0.00</td>
      <td id="T_0aae8_row19_col25" class="data row19 col25" >0.00</td>
      <td id="T_0aae8_row19_col26" class="data row19 col26" >0.00</td>
      <td id="T_0aae8_row19_col27" class="data row19 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row20" class="row_heading level0 row20" >20</th>
      <td id="T_0aae8_row20_col0" class="data row20 col0" >0.00</td>
      <td id="T_0aae8_row20_col1" class="data row20 col1" >0.00</td>
      <td id="T_0aae8_row20_col2" class="data row20 col2" >0.00</td>
      <td id="T_0aae8_row20_col3" class="data row20 col3" >0.00</td>
      <td id="T_0aae8_row20_col4" class="data row20 col4" >0.00</td>
      <td id="T_0aae8_row20_col5" class="data row20 col5" >0.00</td>
      <td id="T_0aae8_row20_col6" class="data row20 col6" >0.00</td>
      <td id="T_0aae8_row20_col7" class="data row20 col7" >0.00</td>
      <td id="T_0aae8_row20_col8" class="data row20 col8" >0.00</td>
      <td id="T_0aae8_row20_col9" class="data row20 col9" >0.00</td>
      <td id="T_0aae8_row20_col10" class="data row20 col10" >0.09</td>
      <td id="T_0aae8_row20_col11" class="data row20 col11" >0.45</td>
      <td id="T_0aae8_row20_col12" class="data row20 col12" >0.87</td>
      <td id="T_0aae8_row20_col13" class="data row20 col13" >0.99</td>
      <td id="T_0aae8_row20_col14" class="data row20 col14" >0.99</td>
      <td id="T_0aae8_row20_col15" class="data row20 col15" >0.99</td>
      <td id="T_0aae8_row20_col16" class="data row20 col16" >0.99</td>
      <td id="T_0aae8_row20_col17" class="data row20 col17" >0.79</td>
      <td id="T_0aae8_row20_col18" class="data row20 col18" >0.31</td>
      <td id="T_0aae8_row20_col19" class="data row20 col19" >0.00</td>
      <td id="T_0aae8_row20_col20" class="data row20 col20" >0.00</td>
      <td id="T_0aae8_row20_col21" class="data row20 col21" >0.00</td>
      <td id="T_0aae8_row20_col22" class="data row20 col22" >0.00</td>
      <td id="T_0aae8_row20_col23" class="data row20 col23" >0.00</td>
      <td id="T_0aae8_row20_col24" class="data row20 col24" >0.00</td>
      <td id="T_0aae8_row20_col25" class="data row20 col25" >0.00</td>
      <td id="T_0aae8_row20_col26" class="data row20 col26" >0.00</td>
      <td id="T_0aae8_row20_col27" class="data row20 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row21" class="row_heading level0 row21" >21</th>
      <td id="T_0aae8_row21_col0" class="data row21 col0" >0.00</td>
      <td id="T_0aae8_row21_col1" class="data row21 col1" >0.00</td>
      <td id="T_0aae8_row21_col2" class="data row21 col2" >0.00</td>
      <td id="T_0aae8_row21_col3" class="data row21 col3" >0.00</td>
      <td id="T_0aae8_row21_col4" class="data row21 col4" >0.00</td>
      <td id="T_0aae8_row21_col5" class="data row21 col5" >0.00</td>
      <td id="T_0aae8_row21_col6" class="data row21 col6" >0.00</td>
      <td id="T_0aae8_row21_col7" class="data row21 col7" >0.00</td>
      <td id="T_0aae8_row21_col8" class="data row21 col8" >0.09</td>
      <td id="T_0aae8_row21_col9" class="data row21 col9" >0.26</td>
      <td id="T_0aae8_row21_col10" class="data row21 col10" >0.84</td>
      <td id="T_0aae8_row21_col11" class="data row21 col11" >0.99</td>
      <td id="T_0aae8_row21_col12" class="data row21 col12" >0.99</td>
      <td id="T_0aae8_row21_col13" class="data row21 col13" >0.99</td>
      <td id="T_0aae8_row21_col14" class="data row21 col14" >0.99</td>
      <td id="T_0aae8_row21_col15" class="data row21 col15" >0.78</td>
      <td id="T_0aae8_row21_col16" class="data row21 col16" >0.32</td>
      <td id="T_0aae8_row21_col17" class="data row21 col17" >0.01</td>
      <td id="T_0aae8_row21_col18" class="data row21 col18" >0.00</td>
      <td id="T_0aae8_row21_col19" class="data row21 col19" >0.00</td>
      <td id="T_0aae8_row21_col20" class="data row21 col20" >0.00</td>
      <td id="T_0aae8_row21_col21" class="data row21 col21" >0.00</td>
      <td id="T_0aae8_row21_col22" class="data row21 col22" >0.00</td>
      <td id="T_0aae8_row21_col23" class="data row21 col23" >0.00</td>
      <td id="T_0aae8_row21_col24" class="data row21 col24" >0.00</td>
      <td id="T_0aae8_row21_col25" class="data row21 col25" >0.00</td>
      <td id="T_0aae8_row21_col26" class="data row21 col26" >0.00</td>
      <td id="T_0aae8_row21_col27" class="data row21 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row22" class="row_heading level0 row22" >22</th>
      <td id="T_0aae8_row22_col0" class="data row22 col0" >0.00</td>
      <td id="T_0aae8_row22_col1" class="data row22 col1" >0.00</td>
      <td id="T_0aae8_row22_col2" class="data row22 col2" >0.00</td>
      <td id="T_0aae8_row22_col3" class="data row22 col3" >0.00</td>
      <td id="T_0aae8_row22_col4" class="data row22 col4" >0.00</td>
      <td id="T_0aae8_row22_col5" class="data row22 col5" >0.00</td>
      <td id="T_0aae8_row22_col6" class="data row22 col6" >0.07</td>
      <td id="T_0aae8_row22_col7" class="data row22 col7" >0.67</td>
      <td id="T_0aae8_row22_col8" class="data row22 col8" >0.86</td>
      <td id="T_0aae8_row22_col9" class="data row22 col9" >0.99</td>
      <td id="T_0aae8_row22_col10" class="data row22 col10" >0.99</td>
      <td id="T_0aae8_row22_col11" class="data row22 col11" >0.99</td>
      <td id="T_0aae8_row22_col12" class="data row22 col12" >0.99</td>
      <td id="T_0aae8_row22_col13" class="data row22 col13" >0.76</td>
      <td id="T_0aae8_row22_col14" class="data row22 col14" >0.31</td>
      <td id="T_0aae8_row22_col15" class="data row22 col15" >0.04</td>
      <td id="T_0aae8_row22_col16" class="data row22 col16" >0.00</td>
      <td id="T_0aae8_row22_col17" class="data row22 col17" >0.00</td>
      <td id="T_0aae8_row22_col18" class="data row22 col18" >0.00</td>
      <td id="T_0aae8_row22_col19" class="data row22 col19" >0.00</td>
      <td id="T_0aae8_row22_col20" class="data row22 col20" >0.00</td>
      <td id="T_0aae8_row22_col21" class="data row22 col21" >0.00</td>
      <td id="T_0aae8_row22_col22" class="data row22 col22" >0.00</td>
      <td id="T_0aae8_row22_col23" class="data row22 col23" >0.00</td>
      <td id="T_0aae8_row22_col24" class="data row22 col24" >0.00</td>
      <td id="T_0aae8_row22_col25" class="data row22 col25" >0.00</td>
      <td id="T_0aae8_row22_col26" class="data row22 col26" >0.00</td>
      <td id="T_0aae8_row22_col27" class="data row22 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row23" class="row_heading level0 row23" >23</th>
      <td id="T_0aae8_row23_col0" class="data row23 col0" >0.00</td>
      <td id="T_0aae8_row23_col1" class="data row23 col1" >0.00</td>
      <td id="T_0aae8_row23_col2" class="data row23 col2" >0.00</td>
      <td id="T_0aae8_row23_col3" class="data row23 col3" >0.00</td>
      <td id="T_0aae8_row23_col4" class="data row23 col4" >0.22</td>
      <td id="T_0aae8_row23_col5" class="data row23 col5" >0.67</td>
      <td id="T_0aae8_row23_col6" class="data row23 col6" >0.89</td>
      <td id="T_0aae8_row23_col7" class="data row23 col7" >0.99</td>
      <td id="T_0aae8_row23_col8" class="data row23 col8" >0.99</td>
      <td id="T_0aae8_row23_col9" class="data row23 col9" >0.99</td>
      <td id="T_0aae8_row23_col10" class="data row23 col10" >0.99</td>
      <td id="T_0aae8_row23_col11" class="data row23 col11" >0.96</td>
      <td id="T_0aae8_row23_col12" class="data row23 col12" >0.52</td>
      <td id="T_0aae8_row23_col13" class="data row23 col13" >0.04</td>
      <td id="T_0aae8_row23_col14" class="data row23 col14" >0.00</td>
      <td id="T_0aae8_row23_col15" class="data row23 col15" >0.00</td>
      <td id="T_0aae8_row23_col16" class="data row23 col16" >0.00</td>
      <td id="T_0aae8_row23_col17" class="data row23 col17" >0.00</td>
      <td id="T_0aae8_row23_col18" class="data row23 col18" >0.00</td>
      <td id="T_0aae8_row23_col19" class="data row23 col19" >0.00</td>
      <td id="T_0aae8_row23_col20" class="data row23 col20" >0.00</td>
      <td id="T_0aae8_row23_col21" class="data row23 col21" >0.00</td>
      <td id="T_0aae8_row23_col22" class="data row23 col22" >0.00</td>
      <td id="T_0aae8_row23_col23" class="data row23 col23" >0.00</td>
      <td id="T_0aae8_row23_col24" class="data row23 col24" >0.00</td>
      <td id="T_0aae8_row23_col25" class="data row23 col25" >0.00</td>
      <td id="T_0aae8_row23_col26" class="data row23 col26" >0.00</td>
      <td id="T_0aae8_row23_col27" class="data row23 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row24" class="row_heading level0 row24" >24</th>
      <td id="T_0aae8_row24_col0" class="data row24 col0" >0.00</td>
      <td id="T_0aae8_row24_col1" class="data row24 col1" >0.00</td>
      <td id="T_0aae8_row24_col2" class="data row24 col2" >0.00</td>
      <td id="T_0aae8_row24_col3" class="data row24 col3" >0.00</td>
      <td id="T_0aae8_row24_col4" class="data row24 col4" >0.53</td>
      <td id="T_0aae8_row24_col5" class="data row24 col5" >0.99</td>
      <td id="T_0aae8_row24_col6" class="data row24 col6" >0.99</td>
      <td id="T_0aae8_row24_col7" class="data row24 col7" >0.99</td>
      <td id="T_0aae8_row24_col8" class="data row24 col8" >0.83</td>
      <td id="T_0aae8_row24_col9" class="data row24 col9" >0.53</td>
      <td id="T_0aae8_row24_col10" class="data row24 col10" >0.52</td>
      <td id="T_0aae8_row24_col11" class="data row24 col11" >0.06</td>
      <td id="T_0aae8_row24_col12" class="data row24 col12" >0.00</td>
      <td id="T_0aae8_row24_col13" class="data row24 col13" >0.00</td>
      <td id="T_0aae8_row24_col14" class="data row24 col14" >0.00</td>
      <td id="T_0aae8_row24_col15" class="data row24 col15" >0.00</td>
      <td id="T_0aae8_row24_col16" class="data row24 col16" >0.00</td>
      <td id="T_0aae8_row24_col17" class="data row24 col17" >0.00</td>
      <td id="T_0aae8_row24_col18" class="data row24 col18" >0.00</td>
      <td id="T_0aae8_row24_col19" class="data row24 col19" >0.00</td>
      <td id="T_0aae8_row24_col20" class="data row24 col20" >0.00</td>
      <td id="T_0aae8_row24_col21" class="data row24 col21" >0.00</td>
      <td id="T_0aae8_row24_col22" class="data row24 col22" >0.00</td>
      <td id="T_0aae8_row24_col23" class="data row24 col23" >0.00</td>
      <td id="T_0aae8_row24_col24" class="data row24 col24" >0.00</td>
      <td id="T_0aae8_row24_col25" class="data row24 col25" >0.00</td>
      <td id="T_0aae8_row24_col26" class="data row24 col26" >0.00</td>
      <td id="T_0aae8_row24_col27" class="data row24 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row25" class="row_heading level0 row25" >25</th>
      <td id="T_0aae8_row25_col0" class="data row25 col0" >0.00</td>
      <td id="T_0aae8_row25_col1" class="data row25 col1" >0.00</td>
      <td id="T_0aae8_row25_col2" class="data row25 col2" >0.00</td>
      <td id="T_0aae8_row25_col3" class="data row25 col3" >0.00</td>
      <td id="T_0aae8_row25_col4" class="data row25 col4" >0.00</td>
      <td id="T_0aae8_row25_col5" class="data row25 col5" >0.00</td>
      <td id="T_0aae8_row25_col6" class="data row25 col6" >0.00</td>
      <td id="T_0aae8_row25_col7" class="data row25 col7" >0.00</td>
      <td id="T_0aae8_row25_col8" class="data row25 col8" >0.00</td>
      <td id="T_0aae8_row25_col9" class="data row25 col9" >0.00</td>
      <td id="T_0aae8_row25_col10" class="data row25 col10" >0.00</td>
      <td id="T_0aae8_row25_col11" class="data row25 col11" >0.00</td>
      <td id="T_0aae8_row25_col12" class="data row25 col12" >0.00</td>
      <td id="T_0aae8_row25_col13" class="data row25 col13" >0.00</td>
      <td id="T_0aae8_row25_col14" class="data row25 col14" >0.00</td>
      <td id="T_0aae8_row25_col15" class="data row25 col15" >0.00</td>
      <td id="T_0aae8_row25_col16" class="data row25 col16" >0.00</td>
      <td id="T_0aae8_row25_col17" class="data row25 col17" >0.00</td>
      <td id="T_0aae8_row25_col18" class="data row25 col18" >0.00</td>
      <td id="T_0aae8_row25_col19" class="data row25 col19" >0.00</td>
      <td id="T_0aae8_row25_col20" class="data row25 col20" >0.00</td>
      <td id="T_0aae8_row25_col21" class="data row25 col21" >0.00</td>
      <td id="T_0aae8_row25_col22" class="data row25 col22" >0.00</td>
      <td id="T_0aae8_row25_col23" class="data row25 col23" >0.00</td>
      <td id="T_0aae8_row25_col24" class="data row25 col24" >0.00</td>
      <td id="T_0aae8_row25_col25" class="data row25 col25" >0.00</td>
      <td id="T_0aae8_row25_col26" class="data row25 col26" >0.00</td>
      <td id="T_0aae8_row25_col27" class="data row25 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row26" class="row_heading level0 row26" >26</th>
      <td id="T_0aae8_row26_col0" class="data row26 col0" >0.00</td>
      <td id="T_0aae8_row26_col1" class="data row26 col1" >0.00</td>
      <td id="T_0aae8_row26_col2" class="data row26 col2" >0.00</td>
      <td id="T_0aae8_row26_col3" class="data row26 col3" >0.00</td>
      <td id="T_0aae8_row26_col4" class="data row26 col4" >0.00</td>
      <td id="T_0aae8_row26_col5" class="data row26 col5" >0.00</td>
      <td id="T_0aae8_row26_col6" class="data row26 col6" >0.00</td>
      <td id="T_0aae8_row26_col7" class="data row26 col7" >0.00</td>
      <td id="T_0aae8_row26_col8" class="data row26 col8" >0.00</td>
      <td id="T_0aae8_row26_col9" class="data row26 col9" >0.00</td>
      <td id="T_0aae8_row26_col10" class="data row26 col10" >0.00</td>
      <td id="T_0aae8_row26_col11" class="data row26 col11" >0.00</td>
      <td id="T_0aae8_row26_col12" class="data row26 col12" >0.00</td>
      <td id="T_0aae8_row26_col13" class="data row26 col13" >0.00</td>
      <td id="T_0aae8_row26_col14" class="data row26 col14" >0.00</td>
      <td id="T_0aae8_row26_col15" class="data row26 col15" >0.00</td>
      <td id="T_0aae8_row26_col16" class="data row26 col16" >0.00</td>
      <td id="T_0aae8_row26_col17" class="data row26 col17" >0.00</td>
      <td id="T_0aae8_row26_col18" class="data row26 col18" >0.00</td>
      <td id="T_0aae8_row26_col19" class="data row26 col19" >0.00</td>
      <td id="T_0aae8_row26_col20" class="data row26 col20" >0.00</td>
      <td id="T_0aae8_row26_col21" class="data row26 col21" >0.00</td>
      <td id="T_0aae8_row26_col22" class="data row26 col22" >0.00</td>
      <td id="T_0aae8_row26_col23" class="data row26 col23" >0.00</td>
      <td id="T_0aae8_row26_col24" class="data row26 col24" >0.00</td>
      <td id="T_0aae8_row26_col25" class="data row26 col25" >0.00</td>
      <td id="T_0aae8_row26_col26" class="data row26 col26" >0.00</td>
      <td id="T_0aae8_row26_col27" class="data row26 col27" >0.00</td>
    </tr>
    <tr>
      <th id="T_0aae8_level0_row27" class="row_heading level0 row27" >27</th>
      <td id="T_0aae8_row27_col0" class="data row27 col0" >0.00</td>
      <td id="T_0aae8_row27_col1" class="data row27 col1" >0.00</td>
      <td id="T_0aae8_row27_col2" class="data row27 col2" >0.00</td>
      <td id="T_0aae8_row27_col3" class="data row27 col3" >0.00</td>
      <td id="T_0aae8_row27_col4" class="data row27 col4" >0.00</td>
      <td id="T_0aae8_row27_col5" class="data row27 col5" >0.00</td>
      <td id="T_0aae8_row27_col6" class="data row27 col6" >0.00</td>
      <td id="T_0aae8_row27_col7" class="data row27 col7" >0.00</td>
      <td id="T_0aae8_row27_col8" class="data row27 col8" >0.00</td>
      <td id="T_0aae8_row27_col9" class="data row27 col9" >0.00</td>
      <td id="T_0aae8_row27_col10" class="data row27 col10" >0.00</td>
      <td id="T_0aae8_row27_col11" class="data row27 col11" >0.00</td>
      <td id="T_0aae8_row27_col12" class="data row27 col12" >0.00</td>
      <td id="T_0aae8_row27_col13" class="data row27 col13" >0.00</td>
      <td id="T_0aae8_row27_col14" class="data row27 col14" >0.00</td>
      <td id="T_0aae8_row27_col15" class="data row27 col15" >0.00</td>
      <td id="T_0aae8_row27_col16" class="data row27 col16" >0.00</td>
      <td id="T_0aae8_row27_col17" class="data row27 col17" >0.00</td>
      <td id="T_0aae8_row27_col18" class="data row27 col18" >0.00</td>
      <td id="T_0aae8_row27_col19" class="data row27 col19" >0.00</td>
      <td id="T_0aae8_row27_col20" class="data row27 col20" >0.00</td>
      <td id="T_0aae8_row27_col21" class="data row27 col21" >0.00</td>
      <td id="T_0aae8_row27_col22" class="data row27 col22" >0.00</td>
      <td id="T_0aae8_row27_col23" class="data row27 col23" >0.00</td>
      <td id="T_0aae8_row27_col24" class="data row27 col24" >0.00</td>
      <td id="T_0aae8_row27_col25" class="data row27 col25" >0.00</td>
      <td id="T_0aae8_row27_col26" class="data row27 col26" >0.00</td>
      <td id="T_0aae8_row27_col27" class="data row27 col27" >0.00</td>
    </tr>
  </tbody>
</table>


We can also check the distribution of the dataset, and see that it's approximately uniform across number of samples of each digit.


### Create a validation dataset
It's important to carefully split out a validation set from the training set to use to see how well the model generalizes to unseen data during training.

```python
import numpy as np

validation_frac = 0.2
num_samples = len(train_dataset)
split_idx = int(np.floor((1 - validation_frac) * num_samples))
train_idx = np.arange(split_idx)
valid_idx = np.arange(split_idx, num_samples)

train_data = Subset(train_dataset, train_idx)
valid_data = Subset(train_dataset, valid_idx)

```

### Load Datasets into DataLoaders

`torch.utils.data.Dataset` and `torch.util.data.DataLoader` are the two primitives used in PyTorch to decouple pre-loaded data and data for model training.

Datasets, as we saw above, store inputs and their labels, and its elements can be accessed like with a Python dictionary.

DataLoaders, on the other hand, are more useful when training a model.

> "We typically want to pass samples in “minibatches”, reshuffle the data at every epoch to reduce model overfitting, and use Python’s multiprocessing to speed up data retrieval.
>
> DataLoader is an iterable that abstracts this complexity for us in an easy API."

Source: [PyTorch data tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

One aspect of DataLoaders to keep in mind is the batch size parameter. Batch size can be dependent on dataset and architecture, both in terms of training stability and in terms of memory capacity. A larger batch size results in more stable training since the variance in gradient estimation per batch is reduced. However a larger batch size can cause the GPU to run out of memory.

We start experimenting with a batch size that is the length of the entire dataset and update it below for different approaches as needed. Note for our first baseline that no gradients are computed so batch size has no influence on "convergence speed."


```python
train_dataloader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=len(valid_data), shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
```

### Pixel Similarity baseline model

Now we can start trying to create a model to classify the images as digits. Our first model takes a non-ML approach in favor of a straightforward elementwise pixel to pixel comparison per image. We compare a sample image to a mean image for each digit. The mean image is computed by averaging over the training examples per digit.

First we calculate the mean image per digit.


```python
train_data_by_digit = {}
valid_data_by_digit = {}

for i in range(10):
  train_data_by_digit[i] = \
    torch.stack([sample[0] for sample in Subset(train_dataset, list(filter(lambda x: x < split_idx, digit_indices[i])))])
  valid_data_by_digit[i] = \
    torch.stack([sample[0] for sample in Subset(train_dataset, list(filter(lambda x: x >= split_idx, digit_indices[i])))])

digit_means = {y: torch.Tensor() for y in range(10)}
for i in range(10):
  digit_means[i] = train_data_by_digit[i].mean(axis=0)

for i in range(10):
  plt.subplot(2, 5, i + 1)
  plt.imshow(torch.squeeze(digit_means[i]), cmap=plt.get_cmap('gray'))
plt.show()
```


<img src="/assets/img/mnist/mnist_34_0.png">
    


Next, we can make predictions based on pixel-wise comparisons.


```python
import torch.nn.functional as F

def predict(sample):
  # choose the digit whose mean image is closest to the sample
  return torch.argmin(torch.tensor([F.l1_loss(sample, torch.squeeze(digit_means[i])) for i in range(10)]))

def predict_batch(samples):
  return torch.tensor([predict(torch.squeeze(sample)) for sample in samples])

preds = torch.empty(0)
labels = torch.empty(0)
for batch in test_dataloader:
  images, ls = batch
  preds = torch.cat((preds, predict_batch(images)), dim = 0)
  labels = torch.cat((labels, ls), dim = 0)
```

Lets see our prediction accuracy for this baseline:

```python
accuracy = torch.sum(torch.eq(labels, preds)) / len(labels)
print(accuracy)
```

    tensor(0.6673)

Lets also see the confusion matrix. It's interesting that if guessing incorrectly, it's likely to be guessing the digit 1, which makes some sense!

```python
from sklearn.metrics import confusion_matrix
import seaborn as sn

cf_matrix = confusion_matrix(labels, preds)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in range(10)],
                     columns = [i for i in range(10)])

plt.figure(figsize = (10,10))
sn.heatmap(df_cm, annot=True)
plt.show()
```




<img src="/assets/img/mnist/mnist_40_1.png">
    
I'm surprised by how good this baseline already is, without any ML (>60% classification accuracy), but we can definitely do better.

### Learner class from scratch

There is a 7 step process for iterating on model weights:

1. Initialize params
2. Calculate predictions
3. Calculate the loss
4. Calculate the gradients
5. Step the parameters
6. Repeat the process
7. Stop

Lets implement a class that does this.



```python
class Learner:
  def __init__(self, dataloaders, model, optimizer, loss_func, metric, scheduler=None):
    self.dataloaders = dataloaders
    self.model = model
    self.optimizer = optimizer
    self.loss_func = loss_func
    self.metric = metric
    self.scheduler = scheduler
    self.val_losses = []

  def fit(self, epochs):
    for epoch in range(epochs):
      print("---- epoch: ", epoch, "/", epochs - 1, " ----")

      self.model.train()
      train_loss = 0.
      for (train_features, train_labels) in self.dataloaders.train_dl():
        preds = self.model(train_features)
        loss = self.loss_func(preds, train_labels)
        train_loss += loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scheduler:
          self.scheduler.step()
      print("avg training loss: ", train_loss / len(self.dataloaders.train_dl()))

      self.model.eval()
      with torch.no_grad():
        # We evaluate on the entire validation dataset
        val_preds = []
        val_labels = []
        for (val_features, val_ls) in self.dataloaders.valid_dl():
          val_preds.append(self.model(val_features))
          val_labels.append(val_ls)
        val_preds = torch.squeeze(torch.stack(val_preds, dim=0))
        val_labels = torch.squeeze(torch.stack(val_labels, dim=0))
        val_loss = self.loss_func(val_preds, val_labels)
        print("validation loss: ", val_loss)
        print("metric: ", self.metric(val_preds, val_labels))

        # Early stopping
        self.val_losses.append(val_loss)
        if len(self.val_losses) > 2 and self.val_losses[-1] > self.val_losses[-2] and self.val_losses[-2] > self.val_losses[-3]:
          print("stopping condition met")
          break
```

We also create a class to group together training and validation datasets that our Learner needs.

```python
class DataLoaders:
  def __init__(self, train_dataloader, valid_dataloader):
    self.train_dataloader = train_dataloader
    self.valid_dataloader = valid_dataloader

  def train_dl(self):
    return self.train_dataloader

  def valid_dl(self):
    return self.valid_dataloader
```

### Train a linear model

Now that we have a Learner class, we can train a linear model for sanity checking and to get another baseline. We still need a few concrete components in order to train: an architecture, an optimizer, and a loss function. Additionally, with a Linear model, we need to work with flattened data. We use the flattened dataset that was prepared in the colab notebook, but not shown here.

```python
bs = 64
lr = 1e-1
```
```python
train_dataloader = DataLoader(train_data_flat, batch_size=bs, shuffle=True, drop_last=True)
valid_dataloader = DataLoader(valid_data_flat, batch_size=len(valid_data), shuffle=True)
test_dataloader = DataLoader(test_dataset_flattened, batch_size=len(test_dataset_flattened), shuffle=True)

dls = DataLoaders(train_dataloader, valid_dataloader)
```

This is the metric we'll use to see how well each model is able to classify digits.

```python
def digit_accuracy(preds, labels):
  return (torch.argmax(preds, axis=1) == labels).float().mean()
```

We'll use this same loss function for all our models.

```python
loss_func = torch.nn.CrossEntropyLoss()
```

Now let's construct our model and optimizer, then feed all of them to a `Learner`.
```python
model = torch.nn.Linear(28*28,10)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

learner = Learner(dls, model, optimizer, loss_func, digit_accuracy)

learner.fit(1)
```

    ---- epoch:  0 / 0  ----
    avg training loss:  tensor(0.5105, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.3485)
    metric:  tensor(0.9063)


Let's see the test accuracy.

```python
test_feats, test_labels = next(iter(test_dataloader))
preds = model(test_feats)
print("test accuracy: ", digit_accuracy(preds, test_labels))
```

    test accuracy:  tensor(0.9065)

Cool, looks like our linear model has learned something about handwritten digits and has a big improvement over the pixel-wise comparison baseline. But a linear model can only learn so much; a nonlinear model has more wiggle room (pun-intended) to fit the data.

### Train a feed-forward network model


```python
ffn_model = torch.nn.Sequential(torch.nn.Linear(28*28, 64),
                                torch.nn.ReLU(),
                                torch.nn.Linear(64, 10)
                               )

lr = 1e-1
ffn_optimizer = torch.optim.SGD(ffn_model.parameters(), lr=lr)

learner = Learner(dls, ffn_model, ffn_optimizer, loss_func, digit_accuracy)

learner.fit(10)
```

    ---- epoch:  0 / 9  ----
    avg training loss:  tensor(0.4878, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.2750)
    metric:  tensor(0.9197)
    ---- epoch:  1 / 9  ----
    avg training loss:  tensor(0.2544, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.2246)
    metric:  tensor(0.9354)
    ---- epoch:  2 / 9  ----
    avg training loss:  tensor(0.2006, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.1842)
    metric:  tensor(0.9498)
    ---- epoch:  3 / 9  ----
    avg training loss:  tensor(0.1664, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.1586)
    metric:  tensor(0.9557)
    ---- epoch:  4 / 9  ----
    avg training loss:  tensor(0.1429, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.1455)
    metric:  tensor(0.9603)
    ---- epoch:  5 / 9  ----
    avg training loss:  tensor(0.1257, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.1348)
    metric:  tensor(0.9604)
    ---- epoch:  6 / 9  ----
    avg training loss:  tensor(0.1118, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.1280)
    metric:  tensor(0.9630)
    ---- epoch:  7 / 9  ----
    avg training loss:  tensor(0.1015, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.1214)
    metric:  tensor(0.9647)
    ---- epoch:  8 / 9  ----
    avg training loss:  tensor(0.0919, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.1220)
    metric:  tensor(0.9635)
    ---- epoch:  9 / 9  ----
    avg training loss:  tensor(0.0843, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.1135)
    metric:  tensor(0.9672)



```python
test_feats, test_labels = next(iter(test_dataloader))
preds = ffn_model(test_feats)
print("test accuracy: ", digit_accuracy(preds, test_labels))
```

    test accuracy:  tensor(0.9701)


That's decent accuracy, but we can do better and use fewer parameters by taking advantage of the spatial structure of an image!

### Train a CNN


```python
from torch import nn

def conv(ni, nf, stride=2, ks=3):
  return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks // 2)
```


```python
simple_cnn_model = nn.Sequential(
        conv(1,8, ks=5),        #14x14
        nn.ReLU(),
        conv(8,16),             #7x7
        nn.ReLU(),
        conv(16,32),             #4x4
        nn.ReLU(),
        conv(32,64),             #2x2
        nn.ReLU(),
        conv(64,10),             #1x1
        nn.Flatten()
        )
simple_cnn_optimizer = torch.optim.SGD(simple_cnn_model.parameters(), lr=1e-2)
```


```python
bs = 128 # larger batch size means more stable training, but fewer opportunities to update parameters

# Use the unflattened data
train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=len(valid_data), shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

dls = DataLoaders(train_dataloader, valid_dataloader)
```


```python
learner = Learner(dls, simple_cnn_model, simple_cnn_optimizer, loss_func, digit_accuracy)
learner.fit(3)
```

    ---- epoch:  0 / 2  ----
    avg training loss:  tensor(2.3014, grad_fn=<DivBackward0>)
    validation loss:  tensor(2.3006)
    metric:  tensor(0.1060)
    ---- epoch:  1 / 2  ----
    avg training loss:  tensor(2.2990, grad_fn=<DivBackward0>)
    validation loss:  tensor(2.2983)
    metric:  tensor(0.1060)
    ---- epoch:  2 / 2  ----
    avg training loss:  tensor(2.2957, grad_fn=<DivBackward0>)
    validation loss:  tensor(2.2936)
    metric:  tensor(0.1060)


Uh oh...the model doesn't train very well...we're going to need a few tricks.

### Train a performant CNN to achieve high accuracy on MNIST

#### Learning rate scheduling: 1cycle training

1cycle LR training allows us to stably use a much higher learning rate than if we had kept a static learning rate through the entire training process. It updates the learning rate after every batch, annealing from some low learning rate up to a maximum learning rate, then back down to a rate much lower than the initial rate.



```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(simple_cnn_optimizer, max_lr=0.06, steps_per_epoch=len(train_dataloader), epochs=10)

learner = Learner(dls, simple_cnn_model, simple_cnn_optimizer, loss_func, digit_accuracy, scheduler)
learner.fit(10)
```

    ---- epoch:  0 / 9  ----
    avg training loss:  tensor(1.3388, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.2145)
    metric:  tensor(0.9356)
    ---- epoch:  1 / 9  ----
    avg training loss:  tensor(0.1783, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.1195)
    metric:  tensor(0.9630)
    ---- epoch:  2 / 9  ----
    avg training loss:  tensor(0.1070, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.1086)
    metric:  tensor(0.9668)
    ---- epoch:  3 / 9  ----
    avg training loss:  tensor(0.0745, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.0813)
    metric:  tensor(0.9746)
    ---- epoch:  4 / 9  ----
    avg training loss:  tensor(0.0585, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.0750)
    metric:  tensor(0.9786)
    ---- epoch:  5 / 9  ----
    avg training loss:  tensor(0.0452, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.0807)
    metric:  tensor(0.9772)
    ---- epoch:  6 / 9  ----
    avg training loss:  tensor(0.0346, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.0608)
    metric:  tensor(0.9822)
    ---- epoch:  7 / 9  ----
    avg training loss:  tensor(0.0232, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.0598)
    metric:  tensor(0.9825)
    ---- epoch:  8 / 9  ----
    avg training loss:  tensor(0.0142, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.0584)
    metric:  tensor(0.9847)
    ---- epoch:  9 / 9  ----
    avg training loss:  tensor(0.0093, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.0596)
    metric:  tensor(0.9846)


And evaluating on the test set:
```python
test_feats, test_labels = next(iter(test_dataloader))
preds = simple_cnn_model(test_feats)
print("test accuracy: ", digit_accuracy(preds, test_labels))
```

    test accuracy:  tensor(0.9862)

We're close to our goal of >99% accuracy, but our metrics show we're plateauing. Let's add another technique in the mix to try to make better use of our neural capacity.

#### Batch Normalization
Batch normalization was invented to address "internal covariate shift," and although the issue being solved is debatable, there's no doubt that batch normalization makes training a CNN much easier. This normalization technique finds a mean and variance for activations in a minibatch, reducing the number of activations that are too large or too small (the exploding/vanishing gradient problem).

Whereas a learning rate scheduler warms up to a higher learning rate, with batch norm we can just start off with a high learning rate. We can also acheive even higher accuracy in fewer iterations. 


```python
cnn_model_with_norm = nn.Sequential(
        conv(1 ,8, ks=5),        #14x14
        nn.BatchNorm2d(8),
        nn.ReLU(),
        conv(8 ,16),             #7x7
        nn.BatchNorm2d(16),
        nn.ReLU(),
        conv(16,32),             #4x4
        nn.BatchNorm2d(32),
        nn.ReLU(),
        conv(32,64),             #2x2
        nn.BatchNorm2d(64),
        nn.ReLU(),
        conv(64,10),             #1x1
        nn.BatchNorm2d(10),
        nn.Flatten()
        )
optimizer = torch.optim.SGD(cnn_model_with_norm.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(train_dataloader), epochs=10)
learner = Learner(dls, cnn_model_with_norm, optimizer, loss_func, digit_accuracy, scheduler)
learner.fit(10)
```

    ---- epoch:  0 / 9  ----
    avg training loss:  tensor(0.3510, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.1048)
    metric:  tensor(0.9712)
    ---- epoch:  1 / 9  ----
    avg training loss:  tensor(0.0883, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.0819)
    metric:  tensor(0.9755)
    ---- epoch:  2 / 9  ----
    avg training loss:  tensor(0.0605, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.0589)
    metric:  tensor(0.9831)
    ---- epoch:  3 / 9  ----
    avg training loss:  tensor(0.0439, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.0440)
    metric:  tensor(0.9859)
    ---- epoch:  4 / 9  ----
    avg training loss:  tensor(0.0331, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.0475)
    metric:  tensor(0.9865)
    ---- epoch:  5 / 9  ----
    avg training loss:  tensor(0.0259, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.0409)
    metric:  tensor(0.9877)
    ---- epoch:  6 / 9  ----
    avg training loss:  tensor(0.0189, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.0351)
    metric:  tensor(0.9893)
    ---- epoch:  7 / 9  ----
    avg training loss:  tensor(0.0136, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.0321)
    metric:  tensor(0.9902)
    ---- epoch:  8 / 9  ----
    avg training loss:  tensor(0.0090, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.0320)
    metric:  tensor(0.9903)
    ---- epoch:  9 / 9  ----
    avg training loss:  tensor(0.0061, grad_fn=<DivBackward0>)
    validation loss:  tensor(0.0317)
    metric:  tensor(0.9905)



```python
test_feats, test_labels = next(iter(test_dataloader))
preds = cnn_model_with_norm(test_feats)
print("test accuracy: ", digit_accuracy(preds, test_labels))
```

    test accuracy:  tensor(0.9924)


That's pretty good classification accuracy! Lets look at a few examples for ourselves and see our classification accuracy with our own eyes.


```python
for i in range(10):
  plt.subplot(2, 5, i + 1)
  plt.title(torch.argmax(preds[i]))
  plt.imshow(torch.squeeze(test_feats[i]), cmap=plt.get_cmap('gray'))
plt.show()
```


<img src="/assets/img/mnist/mnist_74_0.png">
    
Awesome! Looks like we're able to recognize handwritten digits pretty well. On to something more challenging...
