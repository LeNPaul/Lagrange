---
layout: post
title: "ResNet From Scratch"
author: "Henry Chang"
categories: journal
tags: [deeplearning]
image: resnet/resnet_paper_2.png
___

ResNets were considered state-of-the-art CNNs for many tasks in computer vision until the last couple years. In this post, we'll first walk through the paper that introduced the idea of residual networks, then dive deep into implementing our own ResNets from scratch. We'll implement our own `torch.nn.Module`s for each layer and look deep under the hood into manipulating tensors in memory to implement a 2d convolution layer and max pool layer from scratch. Along the way, we'll learn how to load state from a pretrained ResNet into our custom ResNet and classify some images, as well as fine-tune a ResNet using the Fashion MNIST dataset.

You can choose to follow along directly in Colab: 
<a target="_blank" href="https://colab.research.google.com/github/henryjchang/henryjchang.github.io/blob/gh-pages/_notebooks/resnet/resnet_from_scratch.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## ResNet Paper Walk-Through

ResNets were introduced in [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) by Kaiming He, et al. in 2015. The techniques described in the paper were used to achieve 3.57% error
on the ImageNet test set, good for 1st place in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2015 classification task.


#### Motivation

Around that time, there had been breakthroughs in making networks deeper, but only up to a point. Input normalization and batch normalization layers had been introduced to address the vanishing/exploding gradient problem which affected convergence during training. However, after that breakthrough, another problem was discovered: accuracy degradation. This was surprisingly not due to overfitting. As more layers were added, not only would the validation accuracy get worse, the training accuracy would also plateau and then worsen. Some intuition for this surprise is that a deeper network composed of the same layers and weights as a shallower network plus additional identity layers at the end should perform no worse than the shallower network; but this turned out not to be the case in experiments.

<img src="/assets/img/resnet/resnet_paper_1.png">


### Key Idea

The key idea of the paper was that in order to make networks deeper (and therefore produce better results), the layers can be reformulated into layer blocks and using skip connections, learn a residual function relative to the input to the block. The intuition here was that it would be easier to learn a residual driven to zero rather than learn an identity layer that included nonlinear activations. So instead of trying to learn a function $H(x)$, learn $F(x) = H(x) - x$.

<img src="/assets/img/resnet/resnet_paper_2.png">


The authors ran experiments comparing "plain" deep networks with the same networks plus skip connections, or "residual networks" (see image below for architecture comparison). The experiments showed

*   for plain networks, a deeper architecture had worse error rate than a shallower architecture
*   for deeper residual networks, a deeper archtecture had better error rate than a shallower architecture
*   for architectures of the same depth, residual networks had better error rate than plain networks

<img src="/assets/img/resnet/resnet_paper_3.png">


In the image above, on the left is VGG-19, introduced in [Very Deep Convolutional Networks For Large Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf). VGG networks were a large improvement on [AlexNet](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), by splitting large convolution kernels into multiple 3x3 kernels, and won the ILSVRC 2014 classification task.

In the middle is a 34-layer plain network and on the right is the same network with skip connections.

#### Identity vs Projection Shortcuts

In order to add the output of a block ($x$) with the residual from the next block ($F(x,W_i)$), the two dimensions must match (across channel, width, height).

An identity skip connection may be used when the dimensions are already the same size; however, when the dimensions do not match, a linear projection must be applied such that the sum is $F(x,W_i)$ + $W_sx$.

$W_s$ can also be applied to $x$ in the case that dimensions already match but was not found to be essential to help with the degradation problem so it is not applied in the identity mapping for simplicity.

Note the notation used for $W_sx$ is for a linear layer but can be applied for a convolutional layer as well.

Note $F(x,W_i)$ can refer to a block with any number of layers (with activation function in between each layer), but the architectures discussed are with 2 and 3 layers. A single layer would have no extra activation so adding the skip connection would make little difference.

#### Deeper Bottleneck Architectures
<img src="/assets/img/resnet/resnet_paper_4.png">

In order to reduce training time for larger and deeper networks, a bottleneck block with 3 layers is used to replace a block with 2 layers. This bottleneck architecture reduces dimensionality when applying the 3x3 convolution kernels, by downsampling the input and upsampling the output.

### Experiments

The following table lists the various ResNet architectures that were tried, ranging from an 18-layer network up to one with 152 layers. They each share the same initial and final layers. Varying depths of convolutional blocks are used in the middle.

Batch norm is applied after each convolution layer and before activation with ReLU.

Weights are initialized following the [initialization scheme](https://arxiv.org/pdf/1502.01852.pdf) proposed by the same author, Kaiming He.

<img src="/assets/img/resnet/resnet_paper_5.png">

The 18-layer ResNet architecture is compared with a plain 18-layer architecture to demonstrate a baseline for no significant accuracy improvement at that depth (although converges faster). The 34-layer ResNet has a marked improvement over its corresponding 34-layer plain net. In the 50-layer and up architectures, the conv blocks switch to use the 3-layer bottleneck architectures. Deeper networks achieved better performance on ImageNet.

The residuals of layers were found to be smaller than their plain counterparts.

Note the initial size of each image is cropped to 224x224, hence the resulting output sizes.

The final layer is a 1000-d fully-connected layer for classification on the ImageNet dataset.

Additional experiments were run on CIFAR-10 with architectures that follow the same general design up to >1000 layers. The deepest network still had good results, but performed worse than a network 10x shallower, which they attributed to overfitting.

# Architecture Implementation From Scratch

Next, we'll build our own ResNet architecture from scratch. We'll design our code to be flexible to the implementations of the building blocks, whether they come from `torch.nn` or are custom. We'll design the following in this section:

*   `ConvLayer` - Composed of a `Conv2d`, `BatchNorm2d`, and optionally a `ReLU` activation
*   `ConvBlock` and `ConvBlockBottleneck` - Composed of two `ConvLayer`s for the former and three for the latter
*   `ResidualBlock` - Adds together the output of a `ConvBlock` (the residual), and either the input to the block or a projection of the input, depending on if downsampling occurred (via striding).
*   `BlockGroup` - Composed of a sequence of `ResidualBlock`s. Each `BlockGroup` corresponds to a `conv{i}_x` "layer" for `i = {2...5}` in the description of each ResNet architecture.
*   `ResNet`  - Puts together the `BlockGroup`s, preprends the input layers, and appends the output layers. A `ResNet` is constructed by specifying the number of `ResidualBlock`s in each `BlockGroup`, the feature dimensions for each `BlockGroup`, and the strides for the first `ResidualBlock`s for each `BlockGroup`. The input and output feature dimensions of the network are also specified.

We'll then test our architecture implementation by making concrete ResNet34 and ResNet50 models using `torch.nn` layers for `Conv2d`, `BatchNorm2d`, etc., and loading pre-trained weights for these models from PyTorch. If our implementation is correct, we should be able to successfully load the weights and achieve the same results when running our models vs. running the corresponding pretrained models.

Note: Much of the code below, and in the following sections, comes from working through [ARENA 3.0](https://github.com/callummcdougall/ARENA_3.0/tree/main) exercises and checking the solutions, although the extensions for generality of building blocks and bottleneck architectures are solely my design. The functionality for being able to specify the lower-level building blocks was inspired by C++ templates. If someone knows a better way to do this, please let me know!

### Install dependencies


```python
import torchvision.transforms as transforms
import torchvision.models as models
import torch as t

from IPython.display import Image, display
import PIL.Image

# Download ImageNet labels
!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

    --2024-02-05 05:18:48--  https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 10472 (10K) [text/plain]
    Saving to: ‘imagenet_classes.txt’
    
    imagenet_classes.tx 100%[===================>]  10.23K  --.-KB/s    in 0s      
    
    2024-02-05 05:18:49 (129 MB/s) - ‘imagenet_classes.txt’ saved [10472/10472]
    



```python
!pip install einops
import einops
```

    Collecting einops
      Downloading einops-0.7.0-py3-none-any.whl (44 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m44.6/44.6 kB[0m [31m5.3 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: einops
    Successfully installed einops-0.7.0



```python
# Import test images from a google drive folder into `/content/test_images/`
!pip install gdown
!gdown --no-check-certificate --folder https://drive.google.com/drive/folders/1K2ks6JVseLQtNZmmk1iqjUahw7DTG6pq?usp=drive_link
```

    Requirement already satisfied: gdown in /usr/local/lib/python3.10/dist-packages (4.7.3)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from gdown) (3.13.1)
    Requirement already satisfied: requests[socks] in /usr/local/lib/python3.10/dist-packages (from gdown) (2.31.0)
    Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from gdown) (1.16.0)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from gdown) (4.66.1)
    Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.10/dist-packages (from gdown) (4.12.3)
    Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.10/dist-packages (from beautifulsoup4->gdown) (2.5)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests[socks]->gdown) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests[socks]->gdown) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests[socks]->gdown) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests[socks]->gdown) (2023.11.17)
    Requirement already satisfied: PySocks!=1.5.7,>=1.5.6 in /usr/local/lib/python3.10/dist-packages (from requests[socks]->gdown) (1.7.1)
    Retrieving folder contents
    /usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py:1100: InsecureRequestWarning: Unverified HTTPS request is being made to host 'drive.google.com'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#tls-warnings
      warnings.warn(
    Processing file 1Qp_bdptgBVW9rnky-zaAZJs3eoC2aLYh general_sherman_tree.jpg
    Processing file 1bsimKCTv6cADPell_tBRQD07aE-HIZmn golden_gate_bridge.jpg
    Processing file 1teJvsGFcJIHogt5cECreBcobKNgGDAyB golden_retriever_puppy.jpg
    Processing file 1nRhbHc2ZfW8kuZx4KwiWIcxqhZg_kO41 grizzly_bear.jpg
    Processing file 1yXD7oI_4YGuq5qvEcVQDGt0qakCH2x_e muni_train.jpg
    Retrieving folder contents completed
    Building directory structure
    Building directory structure completed
    /usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py:1100: InsecureRequestWarning: Unverified HTTPS request is being made to host 'drive.google.com'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#tls-warnings
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py:1100: InsecureRequestWarning: Unverified HTTPS request is being made to host 'doc-10-68-docs.googleusercontent.com'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#tls-warnings
      warnings.warn(
    Downloading...
    From: https://drive.google.com/uc?id=1Qp_bdptgBVW9rnky-zaAZJs3eoC2aLYh
    To: /content/test_images/general_sherman_tree.jpg
    100% 319k/319k [00:00<00:00, 107MB/s]
    /usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py:1100: InsecureRequestWarning: Unverified HTTPS request is being made to host 'drive.google.com'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#tls-warnings
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py:1100: InsecureRequestWarning: Unverified HTTPS request is being made to host 'doc-0c-68-docs.googleusercontent.com'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#tls-warnings
      warnings.warn(
    Downloading...
    From: https://drive.google.com/uc?id=1bsimKCTv6cADPell_tBRQD07aE-HIZmn
    To: /content/test_images/golden_gate_bridge.jpg
    100% 53.9k/53.9k [00:00<00:00, 77.8MB/s]
    /usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py:1100: InsecureRequestWarning: Unverified HTTPS request is being made to host 'drive.google.com'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#tls-warnings
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py:1100: InsecureRequestWarning: Unverified HTTPS request is being made to host 'doc-0k-68-docs.googleusercontent.com'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#tls-warnings
      warnings.warn(
    Downloading...
    From: https://drive.google.com/uc?id=1teJvsGFcJIHogt5cECreBcobKNgGDAyB
    To: /content/test_images/golden_retriever_puppy.jpg
    100% 232k/232k [00:00<00:00, 37.1MB/s]
    /usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py:1100: InsecureRequestWarning: Unverified HTTPS request is being made to host 'drive.google.com'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#tls-warnings
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py:1100: InsecureRequestWarning: Unverified HTTPS request is being made to host 'doc-0s-68-docs.googleusercontent.com'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#tls-warnings
      warnings.warn(
    Downloading...
    From: https://drive.google.com/uc?id=1nRhbHc2ZfW8kuZx4KwiWIcxqhZg_kO41
    To: /content/test_images/grizzly_bear.jpg
    100% 195k/195k [00:00<00:00, 97.2MB/s]
    /usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py:1100: InsecureRequestWarning: Unverified HTTPS request is being made to host 'drive.google.com'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#tls-warnings
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py:1100: InsecureRequestWarning: Unverified HTTPS request is being made to host 'doc-04-68-docs.googleusercontent.com'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#tls-warnings
      warnings.warn(
    Downloading...
    From: https://drive.google.com/uc?id=1yXD7oI_4YGuq5qvEcVQDGt0qakCH2x_e
    To: /content/test_images/muni_train.jpg
    100% 88.7k/88.7k [00:00<00:00, 46.0MB/s]
    Download completed



```python
# for typehints
!pip install jaxtyping
from jaxtyping import Float, Int
from torch import Tensor
from typing import Union, Tuple, Optional
```

    Collecting jaxtyping
      Downloading jaxtyping-0.2.25-py3-none-any.whl (39 kB)
    Requirement already satisfied: numpy>=1.20.0 in /usr/local/lib/python3.10/dist-packages (from jaxtyping) (1.23.5)
    Collecting typeguard<3,>=2.13.3 (from jaxtyping)
      Downloading typeguard-2.13.3-py3-none-any.whl (17 kB)
    Requirement already satisfied: typing-extensions>=3.7.4.1 in /usr/local/lib/python3.10/dist-packages (from jaxtyping) (4.5.0)
    Installing collected packages: typeguard, jaxtyping
    Successfully installed jaxtyping-0.2.25 typeguard-2.13.3


### Templated ResNet architecture implementation


```python
def Conv2dFactory(Conv2d, BatchNorm2d, ReLU):
  class ConvLayer(t.nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size=3, stride=1, \
                 padding=1, activation=False, bias=False):
      super().__init__()
      self.conv = Conv2d(in_feats, out_feats, kernel_size=kernel_size, \
                         stride=stride, padding=padding, bias=bias)
      self.batchnorm2d = BatchNorm2d(out_feats)
      self.relu = ReLU()
      self.activation = activation

    def forward(self, x):
      out = self.batchnorm2d(self.conv(x))
      if not self.activation:
        return out
      else:
        return self.relu(out)

  return ConvLayer
```


```python
def ConvBlockFactory(ConvLayer, Sequential, bottleneck=False):
  class ConvBlock(t.nn.Module):
    def __init__(self, in_feats, out_feats, middle_feats=None, first_stride=1):
      super().__init__()
      self.conv_block = Sequential(ConvLayer(in_feats, out_feats, \
                                             stride=first_stride, \
                                             activation=True),
                                   ConvLayer(out_feats, out_feats))

    def forward(self, x):
      return self.conv_block(x)


  '''
  Here we make a small deviation from the original paper to follow
  https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
  which says:

  The bottleneck of TorchVision places the stride for downsampling to the second
  3x3 convolution while the original paper places it to the first 1x1 convolution.
  This variant improves the accuracy and is known as ResNet V1.5.
  '''
  class ConvBlockBottleneck(t.nn.Module):
    def __init__(self, in_feats, out_feats, middle_feats, first_stride=1):
      super().__init__()
      self.conv_block = Sequential(ConvLayer(in_feats, middle_feats, \
                                             kernel_size=1, \
                                             #stride=first_stride, \
                                             padding=0, activation=True),
                                   ConvLayer(middle_feats, middle_feats, \
                                             stride=first_stride, \
                                             activation=True),
                                   ConvLayer(middle_feats, out_feats,
                                             kernel_size=1, padding=0))

    def forward(self, x):
      return self.conv_block(x)

  if not bottleneck:
    return ConvBlock
  else:
    return ConvBlockBottleneck
```


```python
def ResidualBlockFactory(ConvBlock, ConvLayer, ReLU):
  class ResidualBlock(t.nn.Module):
    def __init__(self, in_feats, out_feats, middle_feats=None, first_stride=1):
      '''
      For compatibility with the pretrained model, the ConvBlock branch is
      declared first, and the optional projection is declared second.
      '''
      super().__init__()
      self.left = ConvBlock(in_feats, out_feats, middle_feats, first_stride)

      if first_stride > 1 or in_feats != out_feats:
        projector = ConvLayer(in_feats, out_feats, kernel_size=1, \
                              stride=first_stride, padding=0)
        self.right = projector
      else:
        self.right = t.nn.Identity()

      self.relu = ReLU()

    def forward(self, x):
      '''
      x: shape (batch, in_feats, height, width)
      Return: shape (batch, out_feats, height / first_stride, width / first_stride)
      '''
      return self.relu(self.left(x) + self.right(x))

  return ResidualBlock
```


```python
def BlockGroupFactory(ResidualBlock, Sequential):
  class BlockGroup(t.nn.Module):
    def __init__(self, n_blocks, in_feats, out_feats, middle_feats=None, first_stride=1):
      '''
      An n_blocks-long sequence of ResidualBlock where only the first block
      uses the provided stride.
      '''
      super().__init__()
      blocks = [ResidualBlock(in_feats, out_feats, middle_feats=middle_feats, \
                              first_stride=first_stride)] + \
                [ResidualBlock(out_feats, out_feats, middle_feats=middle_feats) \
                  for i in range(n_blocks - 1)]
      self.block_group = Sequential(*blocks)

    def forward(self, x):
      '''
      x: shape (batch, in_feats, height, width)
      Return: shape (batch, out_feats, height / first_stride, width / first_stride)
      '''
      return self.block_group(x)

  return BlockGroup
```


```python
def ResNetFactory(ConvLayer, MaxPool2d, BlockGroup, AveragePool, Flatten, \
                  Linear, Sequential):
  class ResNet(t.nn.Module):
    def __init__(
        self,
        input_channels = 3,
        n_blocks_per_group=[3, 4, 6, 3],
        middle_features_per_group=None,
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        self.n_blocks_per_group = n_blocks_per_group
        self.middle_features_per_group = middle_features_per_group
        self.out_features_per_group = out_features_per_group
        self.first_strides_per_group = first_strides_per_group
        self.n_classes = n_classes

        in_features_per_group = [64] + out_features_per_group[:-1]
        if middle_features_per_group is None:
          middle_features_per_group = [None] * len(in_features_per_group)

        self.input_layers = Sequential(ConvLayer(input_channels, 64, \
                                                 kernel_size=7, stride=2, \
                                                 padding=3, activation=True),
                                       MaxPool2d(kernel_size=3, stride=2, \
                                                 padding=1))

        args_per_group = [n_blocks_per_group, in_features_per_group, \
                out_features_per_group, middle_features_per_group, \
                          first_strides_per_group]

        self.block_groups = Sequential(*(BlockGroup(*args) \
                          for args in zip(*args_per_group)))

        self.output_layers = Sequential(AveragePool(), Flatten(), \
                                        Linear(out_features_per_group[-1], \
                                               n_classes))

    def forward(self, x):
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        '''
        x = self.input_layers(x)
        x = self.block_groups(x)
        x = self.output_layers(x)
        return x

  return ResNet
```

### Make concrete ResNet34 and ResNet50 architectures using PyTorch layers

From our templated ResNet architecture, we can construct any of the five ResNet architectures using `torch.nn` building blocks. Here we'll construct a ResNet34 and a ResNet50 so we can test out both types of `ConvBlock`s. ResNet34 does not have the bottleneck design while ResNet50 does.

All layers have a PyTorch implementation, except a global average pool over all spatial dimensions. We can easily implement that from scratch:


```python
class AveragePool(t.nn.Module):
  def forward(self, x):
    return t.mean(x, dim=(2,3)) # Average over all spatial dims
```

Now we can create a concrete ResNet34 architecture with torch.nn layers and the classes we've defined.


```python
Conv2dLayer = Conv2dFactory(t.nn.Conv2d, t.nn.BatchNorm2d, t.nn.ReLU)
ConvBlock = ConvBlockFactory(Conv2dLayer, t.nn.Sequential)
ResidualBlock = ResidualBlockFactory(ConvBlock, Conv2dLayer, t.nn.ReLU)
BlockGroup = BlockGroupFactory(ResidualBlock, t.nn.Sequential)
ResNet = ResNetFactory(Conv2dLayer, t.nn.MaxPool2d, BlockGroup, AveragePool, \
                       t.nn.Flatten, t.nn.Linear, t.nn.Sequential)

my_resnet34 = ResNet()  # default arguments correspond to ResNet34
```

Similarly, here's a concrete ResNet50 architecture. Recall that ResNet50 and up use the bottleneck design.


```python
BottleneckConvBlock = ConvBlockFactory(Conv2dLayer, t.nn.Sequential, bottleneck=True)
ResidualBlock = ResidualBlockFactory(BottleneckConvBlock, Conv2dLayer, t.nn.ReLU)
BlockGroup = BlockGroupFactory(ResidualBlock, t.nn.Sequential)
ResNet = ResNetFactory(Conv2dLayer, t.nn.MaxPool2d, BlockGroup, AveragePool, \
                       t.nn.Flatten, t.nn.Linear, t.nn.Sequential)

my_resnet50 = ResNet(n_blocks_per_group=[3, 4, 6, 3],
                     middle_features_per_group=[64, 128, 256, 512],
                     out_features_per_group=[256, 512, 1024, 2048])
```

### Test the implementation

#### Load weights from pre-trained ResNet34 and ResNet50 models

##### Helper functions


```python
# Copied verbatim from https://github.com/callummcdougall/ARENA_3.0/blob/main/chapter0_fundamentals/exercises/part2_cnns/solutions.py#L548-L565
def copy_weights(my_resnet, pretrained_resnet):
    '''Copy over the weights of `pretrained_resnet` to your resnet.'''

    # Get the state dictionaries for each model, check they have the same number of parameters & buffers
    mydict = my_resnet.state_dict()
    pretraineddict = pretrained_resnet.state_dict()
    assert len(mydict) == len(pretraineddict), "Mismatching state dictionaries."

    # Define a dictionary mapping the names of your parameters / buffers to their values in the pretrained model
    state_dict_to_load = {
        mykey: pretrainedvalue
        for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(mydict.items(), pretraineddict.items())
    }

    # Load in this dictionary to your model
    my_resnet.load_state_dict(state_dict_to_load)

    return my_resnet
```


```python
# Copied verbatim from https://github.com/callummcdougall/ARENA_3.0/blob/main/chapter0_fundamentals/exercises/part2_cnns/utils.py

import pandas as pd
from IPython.display import display
import numpy as np

def print_param_count(*models, display_df=True, use_state_dict=False):
    """
    display_df: bool
        If true, displays styled dataframe
        if false, returns dataframe

    use_state_dict: bool
        If true, uses model.state_dict() to construct dataframe
            This will include buffers, not just params
        If false, uses model.named_parameters() to construct dataframe
            This misses out buffers (more useful for GPT)
    """
    df_list = []
    gmap_list = []
    for i, model in enumerate(models, start=1):
        print(f"Model {i}, total params = {sum([param.numel() for name, param in model.named_parameters()])}")
        iterator = model.state_dict().items() if use_state_dict else model.named_parameters()
        df = pd.DataFrame([
            {f"name_{i}": name, f"shape_{i}": tuple(param.shape), f"num_params_{i}": param.numel()}
            for name, param in iterator
        ]) if (i == 1) else pd.DataFrame([
            {f"num_params_{i}": param.numel(), f"shape_{i}": tuple(param.shape), f"name_{i}": name}
            for name, param in iterator
        ])
        df_list.append(df)
        gmap_list.append(np.log(df[f"num_params_{i}"]))
    df = df_list[0] if len(df_list) == 1 else pd.concat(df_list, axis=1).fillna(0)
    for i in range(1, len(models) + 1):
        df[f"num_params_{i}"] = df[f"num_params_{i}"].astype(int)
    if len(models) > 1:
        param_counts = [df[f"num_params_{i}"].values.tolist() for i in range(1, len(models) + 1)]
        if all([param_counts[0] == param_counts[i] for i in range(1, len(param_counts))]):
            print("All parameter counts match!")
        else:
            print("Parameter counts don't match up exactly.")
    if display_df:
        s = df.style
        for i in range(1, len(models) + 1):
            s = s.background_gradient(cmap="viridis", subset=[f"num_params_{i}"], gmap=gmap_list[i-1])
        with pd.option_context("display.max_rows", 1000):
            display(s)
    else:
        return df
```

##### Load weights from pretrained models

We can check whether our architecture has the right size by attempting to copy weights from a pretrained model into our model.


```python
pretrained_resnet34 = t.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
my_resnet34 = copy_weights(my_resnet34, pretrained_resnet34)
print_param_count(my_resnet34, pretrained_resnet34)
```

    Using cache found in /root/.cache/torch/hub/pytorch_vision_v0.10.0
    /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet34_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet34_Weights.DEFAULT` to get the most up-to-date weights.
      warnings.warn(msg)


    Model 1, total params = 21797672
    Model 2, total params = 21797672
    All parameter counts match!



<style type="text/css">
#T_f92a8_row0_col2, #T_f92a8_row0_col3 {
  background-color: #238a8d;
  color: #f1f1f1;
}
#T_f92a8_row1_col2, #T_f92a8_row1_col3, #T_f92a8_row2_col2, #T_f92a8_row2_col3, #T_f92a8_row4_col2, #T_f92a8_row4_col3, #T_f92a8_row5_col2, #T_f92a8_row5_col3, #T_f92a8_row7_col2, #T_f92a8_row7_col3, #T_f92a8_row8_col2, #T_f92a8_row8_col3, #T_f92a8_row10_col2, #T_f92a8_row10_col3, #T_f92a8_row11_col2, #T_f92a8_row11_col3, #T_f92a8_row13_col2, #T_f92a8_row13_col3, #T_f92a8_row14_col2, #T_f92a8_row14_col3, #T_f92a8_row16_col2, #T_f92a8_row16_col3, #T_f92a8_row17_col2, #T_f92a8_row17_col3, #T_f92a8_row19_col2, #T_f92a8_row19_col3, #T_f92a8_row20_col2, #T_f92a8_row20_col3 {
  background-color: #440154;
  color: #f1f1f1;
}
#T_f92a8_row3_col2, #T_f92a8_row3_col3, #T_f92a8_row6_col2, #T_f92a8_row6_col3, #T_f92a8_row9_col2, #T_f92a8_row9_col3, #T_f92a8_row12_col2, #T_f92a8_row12_col3, #T_f92a8_row15_col2, #T_f92a8_row15_col3, #T_f92a8_row18_col2, #T_f92a8_row18_col3 {
  background-color: #23a983;
  color: #f1f1f1;
}
#T_f92a8_row21_col2, #T_f92a8_row21_col3 {
  background-color: #37b878;
  color: #f1f1f1;
}
#T_f92a8_row22_col2, #T_f92a8_row22_col3, #T_f92a8_row23_col2, #T_f92a8_row23_col3, #T_f92a8_row25_col2, #T_f92a8_row25_col3, #T_f92a8_row26_col2, #T_f92a8_row26_col3, #T_f92a8_row28_col2, #T_f92a8_row28_col3, #T_f92a8_row29_col2, #T_f92a8_row29_col3, #T_f92a8_row31_col2, #T_f92a8_row31_col3, #T_f92a8_row32_col2, #T_f92a8_row32_col3, #T_f92a8_row34_col2, #T_f92a8_row34_col3, #T_f92a8_row35_col2, #T_f92a8_row35_col3, #T_f92a8_row37_col2, #T_f92a8_row37_col3, #T_f92a8_row38_col2, #T_f92a8_row38_col3, #T_f92a8_row40_col2, #T_f92a8_row40_col3, #T_f92a8_row41_col2, #T_f92a8_row41_col3, #T_f92a8_row43_col2, #T_f92a8_row43_col3, #T_f92a8_row44_col2, #T_f92a8_row44_col3, #T_f92a8_row46_col2, #T_f92a8_row46_col3, #T_f92a8_row47_col2, #T_f92a8_row47_col3 {
  background-color: #48186a;
  color: #f1f1f1;
}
#T_f92a8_row24_col2, #T_f92a8_row24_col3, #T_f92a8_row30_col2, #T_f92a8_row30_col3, #T_f92a8_row33_col2, #T_f92a8_row33_col3, #T_f92a8_row36_col2, #T_f92a8_row36_col3, #T_f92a8_row39_col2, #T_f92a8_row39_col3, #T_f92a8_row42_col2, #T_f92a8_row42_col3, #T_f92a8_row45_col2, #T_f92a8_row45_col3 {
  background-color: #56c667;
  color: #000000;
}
#T_f92a8_row27_col2, #T_f92a8_row27_col3 {
  background-color: #24878e;
  color: #f1f1f1;
}
#T_f92a8_row48_col2, #T_f92a8_row48_col3 {
  background-color: #7cd250;
  color: #000000;
}
#T_f92a8_row49_col2, #T_f92a8_row49_col3, #T_f92a8_row50_col2, #T_f92a8_row50_col3, #T_f92a8_row52_col2, #T_f92a8_row52_col3, #T_f92a8_row53_col2, #T_f92a8_row53_col3, #T_f92a8_row55_col2, #T_f92a8_row55_col3, #T_f92a8_row56_col2, #T_f92a8_row56_col3, #T_f92a8_row58_col2, #T_f92a8_row58_col3, #T_f92a8_row59_col2, #T_f92a8_row59_col3, #T_f92a8_row61_col2, #T_f92a8_row61_col3, #T_f92a8_row62_col2, #T_f92a8_row62_col3, #T_f92a8_row64_col2, #T_f92a8_row64_col3, #T_f92a8_row65_col2, #T_f92a8_row65_col3, #T_f92a8_row67_col2, #T_f92a8_row67_col3, #T_f92a8_row68_col2, #T_f92a8_row68_col3, #T_f92a8_row70_col2, #T_f92a8_row70_col3, #T_f92a8_row71_col2, #T_f92a8_row71_col3, #T_f92a8_row73_col2, #T_f92a8_row73_col3, #T_f92a8_row74_col2, #T_f92a8_row74_col3, #T_f92a8_row76_col2, #T_f92a8_row76_col3, #T_f92a8_row77_col2, #T_f92a8_row77_col3, #T_f92a8_row79_col2, #T_f92a8_row79_col3, #T_f92a8_row80_col2, #T_f92a8_row80_col3, #T_f92a8_row82_col2, #T_f92a8_row82_col3, #T_f92a8_row83_col2, #T_f92a8_row83_col3, #T_f92a8_row85_col2, #T_f92a8_row85_col3, #T_f92a8_row86_col2, #T_f92a8_row86_col3 {
  background-color: #472e7c;
  color: #f1f1f1;
}
#T_f92a8_row51_col2, #T_f92a8_row51_col3, #T_f92a8_row57_col2, #T_f92a8_row57_col3, #T_f92a8_row60_col2, #T_f92a8_row60_col3, #T_f92a8_row63_col2, #T_f92a8_row63_col3, #T_f92a8_row66_col2, #T_f92a8_row66_col3, #T_f92a8_row69_col2, #T_f92a8_row69_col3, #T_f92a8_row72_col2, #T_f92a8_row72_col3, #T_f92a8_row75_col2, #T_f92a8_row75_col3, #T_f92a8_row78_col2, #T_f92a8_row78_col3, #T_f92a8_row81_col2, #T_f92a8_row81_col3, #T_f92a8_row84_col2, #T_f92a8_row84_col3 {
  background-color: #a8db34;
  color: #000000;
}
#T_f92a8_row54_col2, #T_f92a8_row54_col3 {
  background-color: #21a685;
  color: #f1f1f1;
}
#T_f92a8_row87_col2, #T_f92a8_row87_col3 {
  background-color: #d5e21a;
  color: #000000;
}
#T_f92a8_row88_col2, #T_f92a8_row88_col3, #T_f92a8_row89_col2, #T_f92a8_row89_col3, #T_f92a8_row91_col2, #T_f92a8_row91_col3, #T_f92a8_row92_col2, #T_f92a8_row92_col3, #T_f92a8_row94_col2, #T_f92a8_row94_col3, #T_f92a8_row95_col2, #T_f92a8_row95_col3, #T_f92a8_row97_col2, #T_f92a8_row97_col3, #T_f92a8_row98_col2, #T_f92a8_row98_col3, #T_f92a8_row100_col2, #T_f92a8_row100_col3, #T_f92a8_row101_col2, #T_f92a8_row101_col3, #T_f92a8_row103_col2, #T_f92a8_row103_col3, #T_f92a8_row104_col2, #T_f92a8_row104_col3, #T_f92a8_row106_col2, #T_f92a8_row106_col3, #T_f92a8_row107_col2, #T_f92a8_row107_col3 {
  background-color: #414287;
  color: #f1f1f1;
}
#T_f92a8_row90_col2, #T_f92a8_row90_col3, #T_f92a8_row96_col2, #T_f92a8_row96_col3, #T_f92a8_row99_col2, #T_f92a8_row99_col3, #T_f92a8_row102_col2, #T_f92a8_row102_col3, #T_f92a8_row105_col2, #T_f92a8_row105_col3 {
  background-color: #fde725;
  color: #000000;
}
#T_f92a8_row93_col2, #T_f92a8_row93_col3 {
  background-color: #50c46a;
  color: #000000;
}
#T_f92a8_row108_col2, #T_f92a8_row108_col3 {
  background-color: #9dd93b;
  color: #000000;
}
#T_f92a8_row109_col2, #T_f92a8_row109_col3 {
  background-color: #3a548c;
  color: #f1f1f1;
}
</style>
<table id="T_f92a8" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_f92a8_level0_col0" class="col_heading level0 col0" >name_1</th>
      <th id="T_f92a8_level0_col1" class="col_heading level0 col1" >shape_1</th>
      <th id="T_f92a8_level0_col2" class="col_heading level0 col2" >num_params_1</th>
      <th id="T_f92a8_level0_col3" class="col_heading level0 col3" >num_params_2</th>
      <th id="T_f92a8_level0_col4" class="col_heading level0 col4" >shape_2</th>
      <th id="T_f92a8_level0_col5" class="col_heading level0 col5" >name_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_f92a8_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_f92a8_row0_col0" class="data row0 col0" >input_layers.0.conv.weight</td>
      <td id="T_f92a8_row0_col1" class="data row0 col1" >(64, 3, 7, 7)</td>
      <td id="T_f92a8_row0_col2" class="data row0 col2" >9408</td>
      <td id="T_f92a8_row0_col3" class="data row0 col3" >9408</td>
      <td id="T_f92a8_row0_col4" class="data row0 col4" >(64, 3, 7, 7)</td>
      <td id="T_f92a8_row0_col5" class="data row0 col5" >conv1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_f92a8_row1_col0" class="data row1 col0" >input_layers.0.batchnorm2d.weight</td>
      <td id="T_f92a8_row1_col1" class="data row1 col1" >(64,)</td>
      <td id="T_f92a8_row1_col2" class="data row1 col2" >64</td>
      <td id="T_f92a8_row1_col3" class="data row1 col3" >64</td>
      <td id="T_f92a8_row1_col4" class="data row1 col4" >(64,)</td>
      <td id="T_f92a8_row1_col5" class="data row1 col5" >bn1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_f92a8_row2_col0" class="data row2 col0" >input_layers.0.batchnorm2d.bias</td>
      <td id="T_f92a8_row2_col1" class="data row2 col1" >(64,)</td>
      <td id="T_f92a8_row2_col2" class="data row2 col2" >64</td>
      <td id="T_f92a8_row2_col3" class="data row2 col3" >64</td>
      <td id="T_f92a8_row2_col4" class="data row2 col4" >(64,)</td>
      <td id="T_f92a8_row2_col5" class="data row2 col5" >bn1.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_f92a8_row3_col0" class="data row3 col0" >block_groups.0.block_group.0.left.conv_block.0.conv.weight</td>
      <td id="T_f92a8_row3_col1" class="data row3 col1" >(64, 64, 3, 3)</td>
      <td id="T_f92a8_row3_col2" class="data row3 col2" >36864</td>
      <td id="T_f92a8_row3_col3" class="data row3 col3" >36864</td>
      <td id="T_f92a8_row3_col4" class="data row3 col4" >(64, 64, 3, 3)</td>
      <td id="T_f92a8_row3_col5" class="data row3 col5" >layer1.0.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_f92a8_row4_col0" class="data row4 col0" >block_groups.0.block_group.0.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_f92a8_row4_col1" class="data row4 col1" >(64,)</td>
      <td id="T_f92a8_row4_col2" class="data row4 col2" >64</td>
      <td id="T_f92a8_row4_col3" class="data row4 col3" >64</td>
      <td id="T_f92a8_row4_col4" class="data row4 col4" >(64,)</td>
      <td id="T_f92a8_row4_col5" class="data row4 col5" >layer1.0.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_f92a8_row5_col0" class="data row5 col0" >block_groups.0.block_group.0.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_f92a8_row5_col1" class="data row5 col1" >(64,)</td>
      <td id="T_f92a8_row5_col2" class="data row5 col2" >64</td>
      <td id="T_f92a8_row5_col3" class="data row5 col3" >64</td>
      <td id="T_f92a8_row5_col4" class="data row5 col4" >(64,)</td>
      <td id="T_f92a8_row5_col5" class="data row5 col5" >layer1.0.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_f92a8_row6_col0" class="data row6 col0" >block_groups.0.block_group.0.left.conv_block.1.conv.weight</td>
      <td id="T_f92a8_row6_col1" class="data row6 col1" >(64, 64, 3, 3)</td>
      <td id="T_f92a8_row6_col2" class="data row6 col2" >36864</td>
      <td id="T_f92a8_row6_col3" class="data row6 col3" >36864</td>
      <td id="T_f92a8_row6_col4" class="data row6 col4" >(64, 64, 3, 3)</td>
      <td id="T_f92a8_row6_col5" class="data row6 col5" >layer1.0.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_f92a8_row7_col0" class="data row7 col0" >block_groups.0.block_group.0.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_f92a8_row7_col1" class="data row7 col1" >(64,)</td>
      <td id="T_f92a8_row7_col2" class="data row7 col2" >64</td>
      <td id="T_f92a8_row7_col3" class="data row7 col3" >64</td>
      <td id="T_f92a8_row7_col4" class="data row7 col4" >(64,)</td>
      <td id="T_f92a8_row7_col5" class="data row7 col5" >layer1.0.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_f92a8_row8_col0" class="data row8 col0" >block_groups.0.block_group.0.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_f92a8_row8_col1" class="data row8 col1" >(64,)</td>
      <td id="T_f92a8_row8_col2" class="data row8 col2" >64</td>
      <td id="T_f92a8_row8_col3" class="data row8 col3" >64</td>
      <td id="T_f92a8_row8_col4" class="data row8 col4" >(64,)</td>
      <td id="T_f92a8_row8_col5" class="data row8 col5" >layer1.0.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_f92a8_row9_col0" class="data row9 col0" >block_groups.0.block_group.1.left.conv_block.0.conv.weight</td>
      <td id="T_f92a8_row9_col1" class="data row9 col1" >(64, 64, 3, 3)</td>
      <td id="T_f92a8_row9_col2" class="data row9 col2" >36864</td>
      <td id="T_f92a8_row9_col3" class="data row9 col3" >36864</td>
      <td id="T_f92a8_row9_col4" class="data row9 col4" >(64, 64, 3, 3)</td>
      <td id="T_f92a8_row9_col5" class="data row9 col5" >layer1.1.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_f92a8_row10_col0" class="data row10 col0" >block_groups.0.block_group.1.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_f92a8_row10_col1" class="data row10 col1" >(64,)</td>
      <td id="T_f92a8_row10_col2" class="data row10 col2" >64</td>
      <td id="T_f92a8_row10_col3" class="data row10 col3" >64</td>
      <td id="T_f92a8_row10_col4" class="data row10 col4" >(64,)</td>
      <td id="T_f92a8_row10_col5" class="data row10 col5" >layer1.1.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_f92a8_row11_col0" class="data row11 col0" >block_groups.0.block_group.1.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_f92a8_row11_col1" class="data row11 col1" >(64,)</td>
      <td id="T_f92a8_row11_col2" class="data row11 col2" >64</td>
      <td id="T_f92a8_row11_col3" class="data row11 col3" >64</td>
      <td id="T_f92a8_row11_col4" class="data row11 col4" >(64,)</td>
      <td id="T_f92a8_row11_col5" class="data row11 col5" >layer1.1.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_f92a8_row12_col0" class="data row12 col0" >block_groups.0.block_group.1.left.conv_block.1.conv.weight</td>
      <td id="T_f92a8_row12_col1" class="data row12 col1" >(64, 64, 3, 3)</td>
      <td id="T_f92a8_row12_col2" class="data row12 col2" >36864</td>
      <td id="T_f92a8_row12_col3" class="data row12 col3" >36864</td>
      <td id="T_f92a8_row12_col4" class="data row12 col4" >(64, 64, 3, 3)</td>
      <td id="T_f92a8_row12_col5" class="data row12 col5" >layer1.1.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_f92a8_row13_col0" class="data row13 col0" >block_groups.0.block_group.1.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_f92a8_row13_col1" class="data row13 col1" >(64,)</td>
      <td id="T_f92a8_row13_col2" class="data row13 col2" >64</td>
      <td id="T_f92a8_row13_col3" class="data row13 col3" >64</td>
      <td id="T_f92a8_row13_col4" class="data row13 col4" >(64,)</td>
      <td id="T_f92a8_row13_col5" class="data row13 col5" >layer1.1.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_f92a8_row14_col0" class="data row14 col0" >block_groups.0.block_group.1.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_f92a8_row14_col1" class="data row14 col1" >(64,)</td>
      <td id="T_f92a8_row14_col2" class="data row14 col2" >64</td>
      <td id="T_f92a8_row14_col3" class="data row14 col3" >64</td>
      <td id="T_f92a8_row14_col4" class="data row14 col4" >(64,)</td>
      <td id="T_f92a8_row14_col5" class="data row14 col5" >layer1.1.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_f92a8_row15_col0" class="data row15 col0" >block_groups.0.block_group.2.left.conv_block.0.conv.weight</td>
      <td id="T_f92a8_row15_col1" class="data row15 col1" >(64, 64, 3, 3)</td>
      <td id="T_f92a8_row15_col2" class="data row15 col2" >36864</td>
      <td id="T_f92a8_row15_col3" class="data row15 col3" >36864</td>
      <td id="T_f92a8_row15_col4" class="data row15 col4" >(64, 64, 3, 3)</td>
      <td id="T_f92a8_row15_col5" class="data row15 col5" >layer1.2.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_f92a8_row16_col0" class="data row16 col0" >block_groups.0.block_group.2.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_f92a8_row16_col1" class="data row16 col1" >(64,)</td>
      <td id="T_f92a8_row16_col2" class="data row16 col2" >64</td>
      <td id="T_f92a8_row16_col3" class="data row16 col3" >64</td>
      <td id="T_f92a8_row16_col4" class="data row16 col4" >(64,)</td>
      <td id="T_f92a8_row16_col5" class="data row16 col5" >layer1.2.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_f92a8_row17_col0" class="data row17 col0" >block_groups.0.block_group.2.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_f92a8_row17_col1" class="data row17 col1" >(64,)</td>
      <td id="T_f92a8_row17_col2" class="data row17 col2" >64</td>
      <td id="T_f92a8_row17_col3" class="data row17 col3" >64</td>
      <td id="T_f92a8_row17_col4" class="data row17 col4" >(64,)</td>
      <td id="T_f92a8_row17_col5" class="data row17 col5" >layer1.2.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_f92a8_row18_col0" class="data row18 col0" >block_groups.0.block_group.2.left.conv_block.1.conv.weight</td>
      <td id="T_f92a8_row18_col1" class="data row18 col1" >(64, 64, 3, 3)</td>
      <td id="T_f92a8_row18_col2" class="data row18 col2" >36864</td>
      <td id="T_f92a8_row18_col3" class="data row18 col3" >36864</td>
      <td id="T_f92a8_row18_col4" class="data row18 col4" >(64, 64, 3, 3)</td>
      <td id="T_f92a8_row18_col5" class="data row18 col5" >layer1.2.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_f92a8_row19_col0" class="data row19 col0" >block_groups.0.block_group.2.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_f92a8_row19_col1" class="data row19 col1" >(64,)</td>
      <td id="T_f92a8_row19_col2" class="data row19 col2" >64</td>
      <td id="T_f92a8_row19_col3" class="data row19 col3" >64</td>
      <td id="T_f92a8_row19_col4" class="data row19 col4" >(64,)</td>
      <td id="T_f92a8_row19_col5" class="data row19 col5" >layer1.2.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row20" class="row_heading level0 row20" >20</th>
      <td id="T_f92a8_row20_col0" class="data row20 col0" >block_groups.0.block_group.2.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_f92a8_row20_col1" class="data row20 col1" >(64,)</td>
      <td id="T_f92a8_row20_col2" class="data row20 col2" >64</td>
      <td id="T_f92a8_row20_col3" class="data row20 col3" >64</td>
      <td id="T_f92a8_row20_col4" class="data row20 col4" >(64,)</td>
      <td id="T_f92a8_row20_col5" class="data row20 col5" >layer1.2.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row21" class="row_heading level0 row21" >21</th>
      <td id="T_f92a8_row21_col0" class="data row21 col0" >block_groups.1.block_group.0.left.conv_block.0.conv.weight</td>
      <td id="T_f92a8_row21_col1" class="data row21 col1" >(128, 64, 3, 3)</td>
      <td id="T_f92a8_row21_col2" class="data row21 col2" >73728</td>
      <td id="T_f92a8_row21_col3" class="data row21 col3" >73728</td>
      <td id="T_f92a8_row21_col4" class="data row21 col4" >(128, 64, 3, 3)</td>
      <td id="T_f92a8_row21_col5" class="data row21 col5" >layer2.0.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row22" class="row_heading level0 row22" >22</th>
      <td id="T_f92a8_row22_col0" class="data row22 col0" >block_groups.1.block_group.0.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_f92a8_row22_col1" class="data row22 col1" >(128,)</td>
      <td id="T_f92a8_row22_col2" class="data row22 col2" >128</td>
      <td id="T_f92a8_row22_col3" class="data row22 col3" >128</td>
      <td id="T_f92a8_row22_col4" class="data row22 col4" >(128,)</td>
      <td id="T_f92a8_row22_col5" class="data row22 col5" >layer2.0.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row23" class="row_heading level0 row23" >23</th>
      <td id="T_f92a8_row23_col0" class="data row23 col0" >block_groups.1.block_group.0.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_f92a8_row23_col1" class="data row23 col1" >(128,)</td>
      <td id="T_f92a8_row23_col2" class="data row23 col2" >128</td>
      <td id="T_f92a8_row23_col3" class="data row23 col3" >128</td>
      <td id="T_f92a8_row23_col4" class="data row23 col4" >(128,)</td>
      <td id="T_f92a8_row23_col5" class="data row23 col5" >layer2.0.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row24" class="row_heading level0 row24" >24</th>
      <td id="T_f92a8_row24_col0" class="data row24 col0" >block_groups.1.block_group.0.left.conv_block.1.conv.weight</td>
      <td id="T_f92a8_row24_col1" class="data row24 col1" >(128, 128, 3, 3)</td>
      <td id="T_f92a8_row24_col2" class="data row24 col2" >147456</td>
      <td id="T_f92a8_row24_col3" class="data row24 col3" >147456</td>
      <td id="T_f92a8_row24_col4" class="data row24 col4" >(128, 128, 3, 3)</td>
      <td id="T_f92a8_row24_col5" class="data row24 col5" >layer2.0.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row25" class="row_heading level0 row25" >25</th>
      <td id="T_f92a8_row25_col0" class="data row25 col0" >block_groups.1.block_group.0.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_f92a8_row25_col1" class="data row25 col1" >(128,)</td>
      <td id="T_f92a8_row25_col2" class="data row25 col2" >128</td>
      <td id="T_f92a8_row25_col3" class="data row25 col3" >128</td>
      <td id="T_f92a8_row25_col4" class="data row25 col4" >(128,)</td>
      <td id="T_f92a8_row25_col5" class="data row25 col5" >layer2.0.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row26" class="row_heading level0 row26" >26</th>
      <td id="T_f92a8_row26_col0" class="data row26 col0" >block_groups.1.block_group.0.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_f92a8_row26_col1" class="data row26 col1" >(128,)</td>
      <td id="T_f92a8_row26_col2" class="data row26 col2" >128</td>
      <td id="T_f92a8_row26_col3" class="data row26 col3" >128</td>
      <td id="T_f92a8_row26_col4" class="data row26 col4" >(128,)</td>
      <td id="T_f92a8_row26_col5" class="data row26 col5" >layer2.0.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row27" class="row_heading level0 row27" >27</th>
      <td id="T_f92a8_row27_col0" class="data row27 col0" >block_groups.1.block_group.0.right.conv.weight</td>
      <td id="T_f92a8_row27_col1" class="data row27 col1" >(128, 64, 1, 1)</td>
      <td id="T_f92a8_row27_col2" class="data row27 col2" >8192</td>
      <td id="T_f92a8_row27_col3" class="data row27 col3" >8192</td>
      <td id="T_f92a8_row27_col4" class="data row27 col4" >(128, 64, 1, 1)</td>
      <td id="T_f92a8_row27_col5" class="data row27 col5" >layer2.0.downsample.0.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row28" class="row_heading level0 row28" >28</th>
      <td id="T_f92a8_row28_col0" class="data row28 col0" >block_groups.1.block_group.0.right.batchnorm2d.weight</td>
      <td id="T_f92a8_row28_col1" class="data row28 col1" >(128,)</td>
      <td id="T_f92a8_row28_col2" class="data row28 col2" >128</td>
      <td id="T_f92a8_row28_col3" class="data row28 col3" >128</td>
      <td id="T_f92a8_row28_col4" class="data row28 col4" >(128,)</td>
      <td id="T_f92a8_row28_col5" class="data row28 col5" >layer2.0.downsample.1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row29" class="row_heading level0 row29" >29</th>
      <td id="T_f92a8_row29_col0" class="data row29 col0" >block_groups.1.block_group.0.right.batchnorm2d.bias</td>
      <td id="T_f92a8_row29_col1" class="data row29 col1" >(128,)</td>
      <td id="T_f92a8_row29_col2" class="data row29 col2" >128</td>
      <td id="T_f92a8_row29_col3" class="data row29 col3" >128</td>
      <td id="T_f92a8_row29_col4" class="data row29 col4" >(128,)</td>
      <td id="T_f92a8_row29_col5" class="data row29 col5" >layer2.0.downsample.1.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row30" class="row_heading level0 row30" >30</th>
      <td id="T_f92a8_row30_col0" class="data row30 col0" >block_groups.1.block_group.1.left.conv_block.0.conv.weight</td>
      <td id="T_f92a8_row30_col1" class="data row30 col1" >(128, 128, 3, 3)</td>
      <td id="T_f92a8_row30_col2" class="data row30 col2" >147456</td>
      <td id="T_f92a8_row30_col3" class="data row30 col3" >147456</td>
      <td id="T_f92a8_row30_col4" class="data row30 col4" >(128, 128, 3, 3)</td>
      <td id="T_f92a8_row30_col5" class="data row30 col5" >layer2.1.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row31" class="row_heading level0 row31" >31</th>
      <td id="T_f92a8_row31_col0" class="data row31 col0" >block_groups.1.block_group.1.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_f92a8_row31_col1" class="data row31 col1" >(128,)</td>
      <td id="T_f92a8_row31_col2" class="data row31 col2" >128</td>
      <td id="T_f92a8_row31_col3" class="data row31 col3" >128</td>
      <td id="T_f92a8_row31_col4" class="data row31 col4" >(128,)</td>
      <td id="T_f92a8_row31_col5" class="data row31 col5" >layer2.1.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row32" class="row_heading level0 row32" >32</th>
      <td id="T_f92a8_row32_col0" class="data row32 col0" >block_groups.1.block_group.1.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_f92a8_row32_col1" class="data row32 col1" >(128,)</td>
      <td id="T_f92a8_row32_col2" class="data row32 col2" >128</td>
      <td id="T_f92a8_row32_col3" class="data row32 col3" >128</td>
      <td id="T_f92a8_row32_col4" class="data row32 col4" >(128,)</td>
      <td id="T_f92a8_row32_col5" class="data row32 col5" >layer2.1.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row33" class="row_heading level0 row33" >33</th>
      <td id="T_f92a8_row33_col0" class="data row33 col0" >block_groups.1.block_group.1.left.conv_block.1.conv.weight</td>
      <td id="T_f92a8_row33_col1" class="data row33 col1" >(128, 128, 3, 3)</td>
      <td id="T_f92a8_row33_col2" class="data row33 col2" >147456</td>
      <td id="T_f92a8_row33_col3" class="data row33 col3" >147456</td>
      <td id="T_f92a8_row33_col4" class="data row33 col4" >(128, 128, 3, 3)</td>
      <td id="T_f92a8_row33_col5" class="data row33 col5" >layer2.1.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row34" class="row_heading level0 row34" >34</th>
      <td id="T_f92a8_row34_col0" class="data row34 col0" >block_groups.1.block_group.1.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_f92a8_row34_col1" class="data row34 col1" >(128,)</td>
      <td id="T_f92a8_row34_col2" class="data row34 col2" >128</td>
      <td id="T_f92a8_row34_col3" class="data row34 col3" >128</td>
      <td id="T_f92a8_row34_col4" class="data row34 col4" >(128,)</td>
      <td id="T_f92a8_row34_col5" class="data row34 col5" >layer2.1.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row35" class="row_heading level0 row35" >35</th>
      <td id="T_f92a8_row35_col0" class="data row35 col0" >block_groups.1.block_group.1.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_f92a8_row35_col1" class="data row35 col1" >(128,)</td>
      <td id="T_f92a8_row35_col2" class="data row35 col2" >128</td>
      <td id="T_f92a8_row35_col3" class="data row35 col3" >128</td>
      <td id="T_f92a8_row35_col4" class="data row35 col4" >(128,)</td>
      <td id="T_f92a8_row35_col5" class="data row35 col5" >layer2.1.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row36" class="row_heading level0 row36" >36</th>
      <td id="T_f92a8_row36_col0" class="data row36 col0" >block_groups.1.block_group.2.left.conv_block.0.conv.weight</td>
      <td id="T_f92a8_row36_col1" class="data row36 col1" >(128, 128, 3, 3)</td>
      <td id="T_f92a8_row36_col2" class="data row36 col2" >147456</td>
      <td id="T_f92a8_row36_col3" class="data row36 col3" >147456</td>
      <td id="T_f92a8_row36_col4" class="data row36 col4" >(128, 128, 3, 3)</td>
      <td id="T_f92a8_row36_col5" class="data row36 col5" >layer2.2.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row37" class="row_heading level0 row37" >37</th>
      <td id="T_f92a8_row37_col0" class="data row37 col0" >block_groups.1.block_group.2.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_f92a8_row37_col1" class="data row37 col1" >(128,)</td>
      <td id="T_f92a8_row37_col2" class="data row37 col2" >128</td>
      <td id="T_f92a8_row37_col3" class="data row37 col3" >128</td>
      <td id="T_f92a8_row37_col4" class="data row37 col4" >(128,)</td>
      <td id="T_f92a8_row37_col5" class="data row37 col5" >layer2.2.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row38" class="row_heading level0 row38" >38</th>
      <td id="T_f92a8_row38_col0" class="data row38 col0" >block_groups.1.block_group.2.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_f92a8_row38_col1" class="data row38 col1" >(128,)</td>
      <td id="T_f92a8_row38_col2" class="data row38 col2" >128</td>
      <td id="T_f92a8_row38_col3" class="data row38 col3" >128</td>
      <td id="T_f92a8_row38_col4" class="data row38 col4" >(128,)</td>
      <td id="T_f92a8_row38_col5" class="data row38 col5" >layer2.2.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row39" class="row_heading level0 row39" >39</th>
      <td id="T_f92a8_row39_col0" class="data row39 col0" >block_groups.1.block_group.2.left.conv_block.1.conv.weight</td>
      <td id="T_f92a8_row39_col1" class="data row39 col1" >(128, 128, 3, 3)</td>
      <td id="T_f92a8_row39_col2" class="data row39 col2" >147456</td>
      <td id="T_f92a8_row39_col3" class="data row39 col3" >147456</td>
      <td id="T_f92a8_row39_col4" class="data row39 col4" >(128, 128, 3, 3)</td>
      <td id="T_f92a8_row39_col5" class="data row39 col5" >layer2.2.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row40" class="row_heading level0 row40" >40</th>
      <td id="T_f92a8_row40_col0" class="data row40 col0" >block_groups.1.block_group.2.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_f92a8_row40_col1" class="data row40 col1" >(128,)</td>
      <td id="T_f92a8_row40_col2" class="data row40 col2" >128</td>
      <td id="T_f92a8_row40_col3" class="data row40 col3" >128</td>
      <td id="T_f92a8_row40_col4" class="data row40 col4" >(128,)</td>
      <td id="T_f92a8_row40_col5" class="data row40 col5" >layer2.2.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row41" class="row_heading level0 row41" >41</th>
      <td id="T_f92a8_row41_col0" class="data row41 col0" >block_groups.1.block_group.2.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_f92a8_row41_col1" class="data row41 col1" >(128,)</td>
      <td id="T_f92a8_row41_col2" class="data row41 col2" >128</td>
      <td id="T_f92a8_row41_col3" class="data row41 col3" >128</td>
      <td id="T_f92a8_row41_col4" class="data row41 col4" >(128,)</td>
      <td id="T_f92a8_row41_col5" class="data row41 col5" >layer2.2.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row42" class="row_heading level0 row42" >42</th>
      <td id="T_f92a8_row42_col0" class="data row42 col0" >block_groups.1.block_group.3.left.conv_block.0.conv.weight</td>
      <td id="T_f92a8_row42_col1" class="data row42 col1" >(128, 128, 3, 3)</td>
      <td id="T_f92a8_row42_col2" class="data row42 col2" >147456</td>
      <td id="T_f92a8_row42_col3" class="data row42 col3" >147456</td>
      <td id="T_f92a8_row42_col4" class="data row42 col4" >(128, 128, 3, 3)</td>
      <td id="T_f92a8_row42_col5" class="data row42 col5" >layer2.3.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row43" class="row_heading level0 row43" >43</th>
      <td id="T_f92a8_row43_col0" class="data row43 col0" >block_groups.1.block_group.3.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_f92a8_row43_col1" class="data row43 col1" >(128,)</td>
      <td id="T_f92a8_row43_col2" class="data row43 col2" >128</td>
      <td id="T_f92a8_row43_col3" class="data row43 col3" >128</td>
      <td id="T_f92a8_row43_col4" class="data row43 col4" >(128,)</td>
      <td id="T_f92a8_row43_col5" class="data row43 col5" >layer2.3.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row44" class="row_heading level0 row44" >44</th>
      <td id="T_f92a8_row44_col0" class="data row44 col0" >block_groups.1.block_group.3.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_f92a8_row44_col1" class="data row44 col1" >(128,)</td>
      <td id="T_f92a8_row44_col2" class="data row44 col2" >128</td>
      <td id="T_f92a8_row44_col3" class="data row44 col3" >128</td>
      <td id="T_f92a8_row44_col4" class="data row44 col4" >(128,)</td>
      <td id="T_f92a8_row44_col5" class="data row44 col5" >layer2.3.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row45" class="row_heading level0 row45" >45</th>
      <td id="T_f92a8_row45_col0" class="data row45 col0" >block_groups.1.block_group.3.left.conv_block.1.conv.weight</td>
      <td id="T_f92a8_row45_col1" class="data row45 col1" >(128, 128, 3, 3)</td>
      <td id="T_f92a8_row45_col2" class="data row45 col2" >147456</td>
      <td id="T_f92a8_row45_col3" class="data row45 col3" >147456</td>
      <td id="T_f92a8_row45_col4" class="data row45 col4" >(128, 128, 3, 3)</td>
      <td id="T_f92a8_row45_col5" class="data row45 col5" >layer2.3.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row46" class="row_heading level0 row46" >46</th>
      <td id="T_f92a8_row46_col0" class="data row46 col0" >block_groups.1.block_group.3.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_f92a8_row46_col1" class="data row46 col1" >(128,)</td>
      <td id="T_f92a8_row46_col2" class="data row46 col2" >128</td>
      <td id="T_f92a8_row46_col3" class="data row46 col3" >128</td>
      <td id="T_f92a8_row46_col4" class="data row46 col4" >(128,)</td>
      <td id="T_f92a8_row46_col5" class="data row46 col5" >layer2.3.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row47" class="row_heading level0 row47" >47</th>
      <td id="T_f92a8_row47_col0" class="data row47 col0" >block_groups.1.block_group.3.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_f92a8_row47_col1" class="data row47 col1" >(128,)</td>
      <td id="T_f92a8_row47_col2" class="data row47 col2" >128</td>
      <td id="T_f92a8_row47_col3" class="data row47 col3" >128</td>
      <td id="T_f92a8_row47_col4" class="data row47 col4" >(128,)</td>
      <td id="T_f92a8_row47_col5" class="data row47 col5" >layer2.3.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row48" class="row_heading level0 row48" >48</th>
      <td id="T_f92a8_row48_col0" class="data row48 col0" >block_groups.2.block_group.0.left.conv_block.0.conv.weight</td>
      <td id="T_f92a8_row48_col1" class="data row48 col1" >(256, 128, 3, 3)</td>
      <td id="T_f92a8_row48_col2" class="data row48 col2" >294912</td>
      <td id="T_f92a8_row48_col3" class="data row48 col3" >294912</td>
      <td id="T_f92a8_row48_col4" class="data row48 col4" >(256, 128, 3, 3)</td>
      <td id="T_f92a8_row48_col5" class="data row48 col5" >layer3.0.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row49" class="row_heading level0 row49" >49</th>
      <td id="T_f92a8_row49_col0" class="data row49 col0" >block_groups.2.block_group.0.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_f92a8_row49_col1" class="data row49 col1" >(256,)</td>
      <td id="T_f92a8_row49_col2" class="data row49 col2" >256</td>
      <td id="T_f92a8_row49_col3" class="data row49 col3" >256</td>
      <td id="T_f92a8_row49_col4" class="data row49 col4" >(256,)</td>
      <td id="T_f92a8_row49_col5" class="data row49 col5" >layer3.0.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row50" class="row_heading level0 row50" >50</th>
      <td id="T_f92a8_row50_col0" class="data row50 col0" >block_groups.2.block_group.0.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_f92a8_row50_col1" class="data row50 col1" >(256,)</td>
      <td id="T_f92a8_row50_col2" class="data row50 col2" >256</td>
      <td id="T_f92a8_row50_col3" class="data row50 col3" >256</td>
      <td id="T_f92a8_row50_col4" class="data row50 col4" >(256,)</td>
      <td id="T_f92a8_row50_col5" class="data row50 col5" >layer3.0.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row51" class="row_heading level0 row51" >51</th>
      <td id="T_f92a8_row51_col0" class="data row51 col0" >block_groups.2.block_group.0.left.conv_block.1.conv.weight</td>
      <td id="T_f92a8_row51_col1" class="data row51 col1" >(256, 256, 3, 3)</td>
      <td id="T_f92a8_row51_col2" class="data row51 col2" >589824</td>
      <td id="T_f92a8_row51_col3" class="data row51 col3" >589824</td>
      <td id="T_f92a8_row51_col4" class="data row51 col4" >(256, 256, 3, 3)</td>
      <td id="T_f92a8_row51_col5" class="data row51 col5" >layer3.0.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row52" class="row_heading level0 row52" >52</th>
      <td id="T_f92a8_row52_col0" class="data row52 col0" >block_groups.2.block_group.0.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_f92a8_row52_col1" class="data row52 col1" >(256,)</td>
      <td id="T_f92a8_row52_col2" class="data row52 col2" >256</td>
      <td id="T_f92a8_row52_col3" class="data row52 col3" >256</td>
      <td id="T_f92a8_row52_col4" class="data row52 col4" >(256,)</td>
      <td id="T_f92a8_row52_col5" class="data row52 col5" >layer3.0.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row53" class="row_heading level0 row53" >53</th>
      <td id="T_f92a8_row53_col0" class="data row53 col0" >block_groups.2.block_group.0.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_f92a8_row53_col1" class="data row53 col1" >(256,)</td>
      <td id="T_f92a8_row53_col2" class="data row53 col2" >256</td>
      <td id="T_f92a8_row53_col3" class="data row53 col3" >256</td>
      <td id="T_f92a8_row53_col4" class="data row53 col4" >(256,)</td>
      <td id="T_f92a8_row53_col5" class="data row53 col5" >layer3.0.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row54" class="row_heading level0 row54" >54</th>
      <td id="T_f92a8_row54_col0" class="data row54 col0" >block_groups.2.block_group.0.right.conv.weight</td>
      <td id="T_f92a8_row54_col1" class="data row54 col1" >(256, 128, 1, 1)</td>
      <td id="T_f92a8_row54_col2" class="data row54 col2" >32768</td>
      <td id="T_f92a8_row54_col3" class="data row54 col3" >32768</td>
      <td id="T_f92a8_row54_col4" class="data row54 col4" >(256, 128, 1, 1)</td>
      <td id="T_f92a8_row54_col5" class="data row54 col5" >layer3.0.downsample.0.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row55" class="row_heading level0 row55" >55</th>
      <td id="T_f92a8_row55_col0" class="data row55 col0" >block_groups.2.block_group.0.right.batchnorm2d.weight</td>
      <td id="T_f92a8_row55_col1" class="data row55 col1" >(256,)</td>
      <td id="T_f92a8_row55_col2" class="data row55 col2" >256</td>
      <td id="T_f92a8_row55_col3" class="data row55 col3" >256</td>
      <td id="T_f92a8_row55_col4" class="data row55 col4" >(256,)</td>
      <td id="T_f92a8_row55_col5" class="data row55 col5" >layer3.0.downsample.1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row56" class="row_heading level0 row56" >56</th>
      <td id="T_f92a8_row56_col0" class="data row56 col0" >block_groups.2.block_group.0.right.batchnorm2d.bias</td>
      <td id="T_f92a8_row56_col1" class="data row56 col1" >(256,)</td>
      <td id="T_f92a8_row56_col2" class="data row56 col2" >256</td>
      <td id="T_f92a8_row56_col3" class="data row56 col3" >256</td>
      <td id="T_f92a8_row56_col4" class="data row56 col4" >(256,)</td>
      <td id="T_f92a8_row56_col5" class="data row56 col5" >layer3.0.downsample.1.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row57" class="row_heading level0 row57" >57</th>
      <td id="T_f92a8_row57_col0" class="data row57 col0" >block_groups.2.block_group.1.left.conv_block.0.conv.weight</td>
      <td id="T_f92a8_row57_col1" class="data row57 col1" >(256, 256, 3, 3)</td>
      <td id="T_f92a8_row57_col2" class="data row57 col2" >589824</td>
      <td id="T_f92a8_row57_col3" class="data row57 col3" >589824</td>
      <td id="T_f92a8_row57_col4" class="data row57 col4" >(256, 256, 3, 3)</td>
      <td id="T_f92a8_row57_col5" class="data row57 col5" >layer3.1.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row58" class="row_heading level0 row58" >58</th>
      <td id="T_f92a8_row58_col0" class="data row58 col0" >block_groups.2.block_group.1.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_f92a8_row58_col1" class="data row58 col1" >(256,)</td>
      <td id="T_f92a8_row58_col2" class="data row58 col2" >256</td>
      <td id="T_f92a8_row58_col3" class="data row58 col3" >256</td>
      <td id="T_f92a8_row58_col4" class="data row58 col4" >(256,)</td>
      <td id="T_f92a8_row58_col5" class="data row58 col5" >layer3.1.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row59" class="row_heading level0 row59" >59</th>
      <td id="T_f92a8_row59_col0" class="data row59 col0" >block_groups.2.block_group.1.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_f92a8_row59_col1" class="data row59 col1" >(256,)</td>
      <td id="T_f92a8_row59_col2" class="data row59 col2" >256</td>
      <td id="T_f92a8_row59_col3" class="data row59 col3" >256</td>
      <td id="T_f92a8_row59_col4" class="data row59 col4" >(256,)</td>
      <td id="T_f92a8_row59_col5" class="data row59 col5" >layer3.1.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row60" class="row_heading level0 row60" >60</th>
      <td id="T_f92a8_row60_col0" class="data row60 col0" >block_groups.2.block_group.1.left.conv_block.1.conv.weight</td>
      <td id="T_f92a8_row60_col1" class="data row60 col1" >(256, 256, 3, 3)</td>
      <td id="T_f92a8_row60_col2" class="data row60 col2" >589824</td>
      <td id="T_f92a8_row60_col3" class="data row60 col3" >589824</td>
      <td id="T_f92a8_row60_col4" class="data row60 col4" >(256, 256, 3, 3)</td>
      <td id="T_f92a8_row60_col5" class="data row60 col5" >layer3.1.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row61" class="row_heading level0 row61" >61</th>
      <td id="T_f92a8_row61_col0" class="data row61 col0" >block_groups.2.block_group.1.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_f92a8_row61_col1" class="data row61 col1" >(256,)</td>
      <td id="T_f92a8_row61_col2" class="data row61 col2" >256</td>
      <td id="T_f92a8_row61_col3" class="data row61 col3" >256</td>
      <td id="T_f92a8_row61_col4" class="data row61 col4" >(256,)</td>
      <td id="T_f92a8_row61_col5" class="data row61 col5" >layer3.1.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row62" class="row_heading level0 row62" >62</th>
      <td id="T_f92a8_row62_col0" class="data row62 col0" >block_groups.2.block_group.1.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_f92a8_row62_col1" class="data row62 col1" >(256,)</td>
      <td id="T_f92a8_row62_col2" class="data row62 col2" >256</td>
      <td id="T_f92a8_row62_col3" class="data row62 col3" >256</td>
      <td id="T_f92a8_row62_col4" class="data row62 col4" >(256,)</td>
      <td id="T_f92a8_row62_col5" class="data row62 col5" >layer3.1.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row63" class="row_heading level0 row63" >63</th>
      <td id="T_f92a8_row63_col0" class="data row63 col0" >block_groups.2.block_group.2.left.conv_block.0.conv.weight</td>
      <td id="T_f92a8_row63_col1" class="data row63 col1" >(256, 256, 3, 3)</td>
      <td id="T_f92a8_row63_col2" class="data row63 col2" >589824</td>
      <td id="T_f92a8_row63_col3" class="data row63 col3" >589824</td>
      <td id="T_f92a8_row63_col4" class="data row63 col4" >(256, 256, 3, 3)</td>
      <td id="T_f92a8_row63_col5" class="data row63 col5" >layer3.2.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row64" class="row_heading level0 row64" >64</th>
      <td id="T_f92a8_row64_col0" class="data row64 col0" >block_groups.2.block_group.2.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_f92a8_row64_col1" class="data row64 col1" >(256,)</td>
      <td id="T_f92a8_row64_col2" class="data row64 col2" >256</td>
      <td id="T_f92a8_row64_col3" class="data row64 col3" >256</td>
      <td id="T_f92a8_row64_col4" class="data row64 col4" >(256,)</td>
      <td id="T_f92a8_row64_col5" class="data row64 col5" >layer3.2.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row65" class="row_heading level0 row65" >65</th>
      <td id="T_f92a8_row65_col0" class="data row65 col0" >block_groups.2.block_group.2.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_f92a8_row65_col1" class="data row65 col1" >(256,)</td>
      <td id="T_f92a8_row65_col2" class="data row65 col2" >256</td>
      <td id="T_f92a8_row65_col3" class="data row65 col3" >256</td>
      <td id="T_f92a8_row65_col4" class="data row65 col4" >(256,)</td>
      <td id="T_f92a8_row65_col5" class="data row65 col5" >layer3.2.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row66" class="row_heading level0 row66" >66</th>
      <td id="T_f92a8_row66_col0" class="data row66 col0" >block_groups.2.block_group.2.left.conv_block.1.conv.weight</td>
      <td id="T_f92a8_row66_col1" class="data row66 col1" >(256, 256, 3, 3)</td>
      <td id="T_f92a8_row66_col2" class="data row66 col2" >589824</td>
      <td id="T_f92a8_row66_col3" class="data row66 col3" >589824</td>
      <td id="T_f92a8_row66_col4" class="data row66 col4" >(256, 256, 3, 3)</td>
      <td id="T_f92a8_row66_col5" class="data row66 col5" >layer3.2.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row67" class="row_heading level0 row67" >67</th>
      <td id="T_f92a8_row67_col0" class="data row67 col0" >block_groups.2.block_group.2.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_f92a8_row67_col1" class="data row67 col1" >(256,)</td>
      <td id="T_f92a8_row67_col2" class="data row67 col2" >256</td>
      <td id="T_f92a8_row67_col3" class="data row67 col3" >256</td>
      <td id="T_f92a8_row67_col4" class="data row67 col4" >(256,)</td>
      <td id="T_f92a8_row67_col5" class="data row67 col5" >layer3.2.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row68" class="row_heading level0 row68" >68</th>
      <td id="T_f92a8_row68_col0" class="data row68 col0" >block_groups.2.block_group.2.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_f92a8_row68_col1" class="data row68 col1" >(256,)</td>
      <td id="T_f92a8_row68_col2" class="data row68 col2" >256</td>
      <td id="T_f92a8_row68_col3" class="data row68 col3" >256</td>
      <td id="T_f92a8_row68_col4" class="data row68 col4" >(256,)</td>
      <td id="T_f92a8_row68_col5" class="data row68 col5" >layer3.2.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row69" class="row_heading level0 row69" >69</th>
      <td id="T_f92a8_row69_col0" class="data row69 col0" >block_groups.2.block_group.3.left.conv_block.0.conv.weight</td>
      <td id="T_f92a8_row69_col1" class="data row69 col1" >(256, 256, 3, 3)</td>
      <td id="T_f92a8_row69_col2" class="data row69 col2" >589824</td>
      <td id="T_f92a8_row69_col3" class="data row69 col3" >589824</td>
      <td id="T_f92a8_row69_col4" class="data row69 col4" >(256, 256, 3, 3)</td>
      <td id="T_f92a8_row69_col5" class="data row69 col5" >layer3.3.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row70" class="row_heading level0 row70" >70</th>
      <td id="T_f92a8_row70_col0" class="data row70 col0" >block_groups.2.block_group.3.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_f92a8_row70_col1" class="data row70 col1" >(256,)</td>
      <td id="T_f92a8_row70_col2" class="data row70 col2" >256</td>
      <td id="T_f92a8_row70_col3" class="data row70 col3" >256</td>
      <td id="T_f92a8_row70_col4" class="data row70 col4" >(256,)</td>
      <td id="T_f92a8_row70_col5" class="data row70 col5" >layer3.3.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row71" class="row_heading level0 row71" >71</th>
      <td id="T_f92a8_row71_col0" class="data row71 col0" >block_groups.2.block_group.3.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_f92a8_row71_col1" class="data row71 col1" >(256,)</td>
      <td id="T_f92a8_row71_col2" class="data row71 col2" >256</td>
      <td id="T_f92a8_row71_col3" class="data row71 col3" >256</td>
      <td id="T_f92a8_row71_col4" class="data row71 col4" >(256,)</td>
      <td id="T_f92a8_row71_col5" class="data row71 col5" >layer3.3.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row72" class="row_heading level0 row72" >72</th>
      <td id="T_f92a8_row72_col0" class="data row72 col0" >block_groups.2.block_group.3.left.conv_block.1.conv.weight</td>
      <td id="T_f92a8_row72_col1" class="data row72 col1" >(256, 256, 3, 3)</td>
      <td id="T_f92a8_row72_col2" class="data row72 col2" >589824</td>
      <td id="T_f92a8_row72_col3" class="data row72 col3" >589824</td>
      <td id="T_f92a8_row72_col4" class="data row72 col4" >(256, 256, 3, 3)</td>
      <td id="T_f92a8_row72_col5" class="data row72 col5" >layer3.3.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row73" class="row_heading level0 row73" >73</th>
      <td id="T_f92a8_row73_col0" class="data row73 col0" >block_groups.2.block_group.3.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_f92a8_row73_col1" class="data row73 col1" >(256,)</td>
      <td id="T_f92a8_row73_col2" class="data row73 col2" >256</td>
      <td id="T_f92a8_row73_col3" class="data row73 col3" >256</td>
      <td id="T_f92a8_row73_col4" class="data row73 col4" >(256,)</td>
      <td id="T_f92a8_row73_col5" class="data row73 col5" >layer3.3.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row74" class="row_heading level0 row74" >74</th>
      <td id="T_f92a8_row74_col0" class="data row74 col0" >block_groups.2.block_group.3.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_f92a8_row74_col1" class="data row74 col1" >(256,)</td>
      <td id="T_f92a8_row74_col2" class="data row74 col2" >256</td>
      <td id="T_f92a8_row74_col3" class="data row74 col3" >256</td>
      <td id="T_f92a8_row74_col4" class="data row74 col4" >(256,)</td>
      <td id="T_f92a8_row74_col5" class="data row74 col5" >layer3.3.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row75" class="row_heading level0 row75" >75</th>
      <td id="T_f92a8_row75_col0" class="data row75 col0" >block_groups.2.block_group.4.left.conv_block.0.conv.weight</td>
      <td id="T_f92a8_row75_col1" class="data row75 col1" >(256, 256, 3, 3)</td>
      <td id="T_f92a8_row75_col2" class="data row75 col2" >589824</td>
      <td id="T_f92a8_row75_col3" class="data row75 col3" >589824</td>
      <td id="T_f92a8_row75_col4" class="data row75 col4" >(256, 256, 3, 3)</td>
      <td id="T_f92a8_row75_col5" class="data row75 col5" >layer3.4.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row76" class="row_heading level0 row76" >76</th>
      <td id="T_f92a8_row76_col0" class="data row76 col0" >block_groups.2.block_group.4.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_f92a8_row76_col1" class="data row76 col1" >(256,)</td>
      <td id="T_f92a8_row76_col2" class="data row76 col2" >256</td>
      <td id="T_f92a8_row76_col3" class="data row76 col3" >256</td>
      <td id="T_f92a8_row76_col4" class="data row76 col4" >(256,)</td>
      <td id="T_f92a8_row76_col5" class="data row76 col5" >layer3.4.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row77" class="row_heading level0 row77" >77</th>
      <td id="T_f92a8_row77_col0" class="data row77 col0" >block_groups.2.block_group.4.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_f92a8_row77_col1" class="data row77 col1" >(256,)</td>
      <td id="T_f92a8_row77_col2" class="data row77 col2" >256</td>
      <td id="T_f92a8_row77_col3" class="data row77 col3" >256</td>
      <td id="T_f92a8_row77_col4" class="data row77 col4" >(256,)</td>
      <td id="T_f92a8_row77_col5" class="data row77 col5" >layer3.4.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row78" class="row_heading level0 row78" >78</th>
      <td id="T_f92a8_row78_col0" class="data row78 col0" >block_groups.2.block_group.4.left.conv_block.1.conv.weight</td>
      <td id="T_f92a8_row78_col1" class="data row78 col1" >(256, 256, 3, 3)</td>
      <td id="T_f92a8_row78_col2" class="data row78 col2" >589824</td>
      <td id="T_f92a8_row78_col3" class="data row78 col3" >589824</td>
      <td id="T_f92a8_row78_col4" class="data row78 col4" >(256, 256, 3, 3)</td>
      <td id="T_f92a8_row78_col5" class="data row78 col5" >layer3.4.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row79" class="row_heading level0 row79" >79</th>
      <td id="T_f92a8_row79_col0" class="data row79 col0" >block_groups.2.block_group.4.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_f92a8_row79_col1" class="data row79 col1" >(256,)</td>
      <td id="T_f92a8_row79_col2" class="data row79 col2" >256</td>
      <td id="T_f92a8_row79_col3" class="data row79 col3" >256</td>
      <td id="T_f92a8_row79_col4" class="data row79 col4" >(256,)</td>
      <td id="T_f92a8_row79_col5" class="data row79 col5" >layer3.4.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row80" class="row_heading level0 row80" >80</th>
      <td id="T_f92a8_row80_col0" class="data row80 col0" >block_groups.2.block_group.4.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_f92a8_row80_col1" class="data row80 col1" >(256,)</td>
      <td id="T_f92a8_row80_col2" class="data row80 col2" >256</td>
      <td id="T_f92a8_row80_col3" class="data row80 col3" >256</td>
      <td id="T_f92a8_row80_col4" class="data row80 col4" >(256,)</td>
      <td id="T_f92a8_row80_col5" class="data row80 col5" >layer3.4.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row81" class="row_heading level0 row81" >81</th>
      <td id="T_f92a8_row81_col0" class="data row81 col0" >block_groups.2.block_group.5.left.conv_block.0.conv.weight</td>
      <td id="T_f92a8_row81_col1" class="data row81 col1" >(256, 256, 3, 3)</td>
      <td id="T_f92a8_row81_col2" class="data row81 col2" >589824</td>
      <td id="T_f92a8_row81_col3" class="data row81 col3" >589824</td>
      <td id="T_f92a8_row81_col4" class="data row81 col4" >(256, 256, 3, 3)</td>
      <td id="T_f92a8_row81_col5" class="data row81 col5" >layer3.5.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row82" class="row_heading level0 row82" >82</th>
      <td id="T_f92a8_row82_col0" class="data row82 col0" >block_groups.2.block_group.5.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_f92a8_row82_col1" class="data row82 col1" >(256,)</td>
      <td id="T_f92a8_row82_col2" class="data row82 col2" >256</td>
      <td id="T_f92a8_row82_col3" class="data row82 col3" >256</td>
      <td id="T_f92a8_row82_col4" class="data row82 col4" >(256,)</td>
      <td id="T_f92a8_row82_col5" class="data row82 col5" >layer3.5.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row83" class="row_heading level0 row83" >83</th>
      <td id="T_f92a8_row83_col0" class="data row83 col0" >block_groups.2.block_group.5.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_f92a8_row83_col1" class="data row83 col1" >(256,)</td>
      <td id="T_f92a8_row83_col2" class="data row83 col2" >256</td>
      <td id="T_f92a8_row83_col3" class="data row83 col3" >256</td>
      <td id="T_f92a8_row83_col4" class="data row83 col4" >(256,)</td>
      <td id="T_f92a8_row83_col5" class="data row83 col5" >layer3.5.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row84" class="row_heading level0 row84" >84</th>
      <td id="T_f92a8_row84_col0" class="data row84 col0" >block_groups.2.block_group.5.left.conv_block.1.conv.weight</td>
      <td id="T_f92a8_row84_col1" class="data row84 col1" >(256, 256, 3, 3)</td>
      <td id="T_f92a8_row84_col2" class="data row84 col2" >589824</td>
      <td id="T_f92a8_row84_col3" class="data row84 col3" >589824</td>
      <td id="T_f92a8_row84_col4" class="data row84 col4" >(256, 256, 3, 3)</td>
      <td id="T_f92a8_row84_col5" class="data row84 col5" >layer3.5.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row85" class="row_heading level0 row85" >85</th>
      <td id="T_f92a8_row85_col0" class="data row85 col0" >block_groups.2.block_group.5.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_f92a8_row85_col1" class="data row85 col1" >(256,)</td>
      <td id="T_f92a8_row85_col2" class="data row85 col2" >256</td>
      <td id="T_f92a8_row85_col3" class="data row85 col3" >256</td>
      <td id="T_f92a8_row85_col4" class="data row85 col4" >(256,)</td>
      <td id="T_f92a8_row85_col5" class="data row85 col5" >layer3.5.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row86" class="row_heading level0 row86" >86</th>
      <td id="T_f92a8_row86_col0" class="data row86 col0" >block_groups.2.block_group.5.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_f92a8_row86_col1" class="data row86 col1" >(256,)</td>
      <td id="T_f92a8_row86_col2" class="data row86 col2" >256</td>
      <td id="T_f92a8_row86_col3" class="data row86 col3" >256</td>
      <td id="T_f92a8_row86_col4" class="data row86 col4" >(256,)</td>
      <td id="T_f92a8_row86_col5" class="data row86 col5" >layer3.5.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row87" class="row_heading level0 row87" >87</th>
      <td id="T_f92a8_row87_col0" class="data row87 col0" >block_groups.3.block_group.0.left.conv_block.0.conv.weight</td>
      <td id="T_f92a8_row87_col1" class="data row87 col1" >(512, 256, 3, 3)</td>
      <td id="T_f92a8_row87_col2" class="data row87 col2" >1179648</td>
      <td id="T_f92a8_row87_col3" class="data row87 col3" >1179648</td>
      <td id="T_f92a8_row87_col4" class="data row87 col4" >(512, 256, 3, 3)</td>
      <td id="T_f92a8_row87_col5" class="data row87 col5" >layer4.0.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row88" class="row_heading level0 row88" >88</th>
      <td id="T_f92a8_row88_col0" class="data row88 col0" >block_groups.3.block_group.0.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_f92a8_row88_col1" class="data row88 col1" >(512,)</td>
      <td id="T_f92a8_row88_col2" class="data row88 col2" >512</td>
      <td id="T_f92a8_row88_col3" class="data row88 col3" >512</td>
      <td id="T_f92a8_row88_col4" class="data row88 col4" >(512,)</td>
      <td id="T_f92a8_row88_col5" class="data row88 col5" >layer4.0.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row89" class="row_heading level0 row89" >89</th>
      <td id="T_f92a8_row89_col0" class="data row89 col0" >block_groups.3.block_group.0.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_f92a8_row89_col1" class="data row89 col1" >(512,)</td>
      <td id="T_f92a8_row89_col2" class="data row89 col2" >512</td>
      <td id="T_f92a8_row89_col3" class="data row89 col3" >512</td>
      <td id="T_f92a8_row89_col4" class="data row89 col4" >(512,)</td>
      <td id="T_f92a8_row89_col5" class="data row89 col5" >layer4.0.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row90" class="row_heading level0 row90" >90</th>
      <td id="T_f92a8_row90_col0" class="data row90 col0" >block_groups.3.block_group.0.left.conv_block.1.conv.weight</td>
      <td id="T_f92a8_row90_col1" class="data row90 col1" >(512, 512, 3, 3)</td>
      <td id="T_f92a8_row90_col2" class="data row90 col2" >2359296</td>
      <td id="T_f92a8_row90_col3" class="data row90 col3" >2359296</td>
      <td id="T_f92a8_row90_col4" class="data row90 col4" >(512, 512, 3, 3)</td>
      <td id="T_f92a8_row90_col5" class="data row90 col5" >layer4.0.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row91" class="row_heading level0 row91" >91</th>
      <td id="T_f92a8_row91_col0" class="data row91 col0" >block_groups.3.block_group.0.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_f92a8_row91_col1" class="data row91 col1" >(512,)</td>
      <td id="T_f92a8_row91_col2" class="data row91 col2" >512</td>
      <td id="T_f92a8_row91_col3" class="data row91 col3" >512</td>
      <td id="T_f92a8_row91_col4" class="data row91 col4" >(512,)</td>
      <td id="T_f92a8_row91_col5" class="data row91 col5" >layer4.0.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row92" class="row_heading level0 row92" >92</th>
      <td id="T_f92a8_row92_col0" class="data row92 col0" >block_groups.3.block_group.0.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_f92a8_row92_col1" class="data row92 col1" >(512,)</td>
      <td id="T_f92a8_row92_col2" class="data row92 col2" >512</td>
      <td id="T_f92a8_row92_col3" class="data row92 col3" >512</td>
      <td id="T_f92a8_row92_col4" class="data row92 col4" >(512,)</td>
      <td id="T_f92a8_row92_col5" class="data row92 col5" >layer4.0.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row93" class="row_heading level0 row93" >93</th>
      <td id="T_f92a8_row93_col0" class="data row93 col0" >block_groups.3.block_group.0.right.conv.weight</td>
      <td id="T_f92a8_row93_col1" class="data row93 col1" >(512, 256, 1, 1)</td>
      <td id="T_f92a8_row93_col2" class="data row93 col2" >131072</td>
      <td id="T_f92a8_row93_col3" class="data row93 col3" >131072</td>
      <td id="T_f92a8_row93_col4" class="data row93 col4" >(512, 256, 1, 1)</td>
      <td id="T_f92a8_row93_col5" class="data row93 col5" >layer4.0.downsample.0.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row94" class="row_heading level0 row94" >94</th>
      <td id="T_f92a8_row94_col0" class="data row94 col0" >block_groups.3.block_group.0.right.batchnorm2d.weight</td>
      <td id="T_f92a8_row94_col1" class="data row94 col1" >(512,)</td>
      <td id="T_f92a8_row94_col2" class="data row94 col2" >512</td>
      <td id="T_f92a8_row94_col3" class="data row94 col3" >512</td>
      <td id="T_f92a8_row94_col4" class="data row94 col4" >(512,)</td>
      <td id="T_f92a8_row94_col5" class="data row94 col5" >layer4.0.downsample.1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row95" class="row_heading level0 row95" >95</th>
      <td id="T_f92a8_row95_col0" class="data row95 col0" >block_groups.3.block_group.0.right.batchnorm2d.bias</td>
      <td id="T_f92a8_row95_col1" class="data row95 col1" >(512,)</td>
      <td id="T_f92a8_row95_col2" class="data row95 col2" >512</td>
      <td id="T_f92a8_row95_col3" class="data row95 col3" >512</td>
      <td id="T_f92a8_row95_col4" class="data row95 col4" >(512,)</td>
      <td id="T_f92a8_row95_col5" class="data row95 col5" >layer4.0.downsample.1.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row96" class="row_heading level0 row96" >96</th>
      <td id="T_f92a8_row96_col0" class="data row96 col0" >block_groups.3.block_group.1.left.conv_block.0.conv.weight</td>
      <td id="T_f92a8_row96_col1" class="data row96 col1" >(512, 512, 3, 3)</td>
      <td id="T_f92a8_row96_col2" class="data row96 col2" >2359296</td>
      <td id="T_f92a8_row96_col3" class="data row96 col3" >2359296</td>
      <td id="T_f92a8_row96_col4" class="data row96 col4" >(512, 512, 3, 3)</td>
      <td id="T_f92a8_row96_col5" class="data row96 col5" >layer4.1.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row97" class="row_heading level0 row97" >97</th>
      <td id="T_f92a8_row97_col0" class="data row97 col0" >block_groups.3.block_group.1.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_f92a8_row97_col1" class="data row97 col1" >(512,)</td>
      <td id="T_f92a8_row97_col2" class="data row97 col2" >512</td>
      <td id="T_f92a8_row97_col3" class="data row97 col3" >512</td>
      <td id="T_f92a8_row97_col4" class="data row97 col4" >(512,)</td>
      <td id="T_f92a8_row97_col5" class="data row97 col5" >layer4.1.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row98" class="row_heading level0 row98" >98</th>
      <td id="T_f92a8_row98_col0" class="data row98 col0" >block_groups.3.block_group.1.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_f92a8_row98_col1" class="data row98 col1" >(512,)</td>
      <td id="T_f92a8_row98_col2" class="data row98 col2" >512</td>
      <td id="T_f92a8_row98_col3" class="data row98 col3" >512</td>
      <td id="T_f92a8_row98_col4" class="data row98 col4" >(512,)</td>
      <td id="T_f92a8_row98_col5" class="data row98 col5" >layer4.1.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row99" class="row_heading level0 row99" >99</th>
      <td id="T_f92a8_row99_col0" class="data row99 col0" >block_groups.3.block_group.1.left.conv_block.1.conv.weight</td>
      <td id="T_f92a8_row99_col1" class="data row99 col1" >(512, 512, 3, 3)</td>
      <td id="T_f92a8_row99_col2" class="data row99 col2" >2359296</td>
      <td id="T_f92a8_row99_col3" class="data row99 col3" >2359296</td>
      <td id="T_f92a8_row99_col4" class="data row99 col4" >(512, 512, 3, 3)</td>
      <td id="T_f92a8_row99_col5" class="data row99 col5" >layer4.1.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row100" class="row_heading level0 row100" >100</th>
      <td id="T_f92a8_row100_col0" class="data row100 col0" >block_groups.3.block_group.1.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_f92a8_row100_col1" class="data row100 col1" >(512,)</td>
      <td id="T_f92a8_row100_col2" class="data row100 col2" >512</td>
      <td id="T_f92a8_row100_col3" class="data row100 col3" >512</td>
      <td id="T_f92a8_row100_col4" class="data row100 col4" >(512,)</td>
      <td id="T_f92a8_row100_col5" class="data row100 col5" >layer4.1.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row101" class="row_heading level0 row101" >101</th>
      <td id="T_f92a8_row101_col0" class="data row101 col0" >block_groups.3.block_group.1.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_f92a8_row101_col1" class="data row101 col1" >(512,)</td>
      <td id="T_f92a8_row101_col2" class="data row101 col2" >512</td>
      <td id="T_f92a8_row101_col3" class="data row101 col3" >512</td>
      <td id="T_f92a8_row101_col4" class="data row101 col4" >(512,)</td>
      <td id="T_f92a8_row101_col5" class="data row101 col5" >layer4.1.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row102" class="row_heading level0 row102" >102</th>
      <td id="T_f92a8_row102_col0" class="data row102 col0" >block_groups.3.block_group.2.left.conv_block.0.conv.weight</td>
      <td id="T_f92a8_row102_col1" class="data row102 col1" >(512, 512, 3, 3)</td>
      <td id="T_f92a8_row102_col2" class="data row102 col2" >2359296</td>
      <td id="T_f92a8_row102_col3" class="data row102 col3" >2359296</td>
      <td id="T_f92a8_row102_col4" class="data row102 col4" >(512, 512, 3, 3)</td>
      <td id="T_f92a8_row102_col5" class="data row102 col5" >layer4.2.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row103" class="row_heading level0 row103" >103</th>
      <td id="T_f92a8_row103_col0" class="data row103 col0" >block_groups.3.block_group.2.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_f92a8_row103_col1" class="data row103 col1" >(512,)</td>
      <td id="T_f92a8_row103_col2" class="data row103 col2" >512</td>
      <td id="T_f92a8_row103_col3" class="data row103 col3" >512</td>
      <td id="T_f92a8_row103_col4" class="data row103 col4" >(512,)</td>
      <td id="T_f92a8_row103_col5" class="data row103 col5" >layer4.2.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row104" class="row_heading level0 row104" >104</th>
      <td id="T_f92a8_row104_col0" class="data row104 col0" >block_groups.3.block_group.2.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_f92a8_row104_col1" class="data row104 col1" >(512,)</td>
      <td id="T_f92a8_row104_col2" class="data row104 col2" >512</td>
      <td id="T_f92a8_row104_col3" class="data row104 col3" >512</td>
      <td id="T_f92a8_row104_col4" class="data row104 col4" >(512,)</td>
      <td id="T_f92a8_row104_col5" class="data row104 col5" >layer4.2.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row105" class="row_heading level0 row105" >105</th>
      <td id="T_f92a8_row105_col0" class="data row105 col0" >block_groups.3.block_group.2.left.conv_block.1.conv.weight</td>
      <td id="T_f92a8_row105_col1" class="data row105 col1" >(512, 512, 3, 3)</td>
      <td id="T_f92a8_row105_col2" class="data row105 col2" >2359296</td>
      <td id="T_f92a8_row105_col3" class="data row105 col3" >2359296</td>
      <td id="T_f92a8_row105_col4" class="data row105 col4" >(512, 512, 3, 3)</td>
      <td id="T_f92a8_row105_col5" class="data row105 col5" >layer4.2.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row106" class="row_heading level0 row106" >106</th>
      <td id="T_f92a8_row106_col0" class="data row106 col0" >block_groups.3.block_group.2.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_f92a8_row106_col1" class="data row106 col1" >(512,)</td>
      <td id="T_f92a8_row106_col2" class="data row106 col2" >512</td>
      <td id="T_f92a8_row106_col3" class="data row106 col3" >512</td>
      <td id="T_f92a8_row106_col4" class="data row106 col4" >(512,)</td>
      <td id="T_f92a8_row106_col5" class="data row106 col5" >layer4.2.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row107" class="row_heading level0 row107" >107</th>
      <td id="T_f92a8_row107_col0" class="data row107 col0" >block_groups.3.block_group.2.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_f92a8_row107_col1" class="data row107 col1" >(512,)</td>
      <td id="T_f92a8_row107_col2" class="data row107 col2" >512</td>
      <td id="T_f92a8_row107_col3" class="data row107 col3" >512</td>
      <td id="T_f92a8_row107_col4" class="data row107 col4" >(512,)</td>
      <td id="T_f92a8_row107_col5" class="data row107 col5" >layer4.2.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row108" class="row_heading level0 row108" >108</th>
      <td id="T_f92a8_row108_col0" class="data row108 col0" >output_layers.2.weight</td>
      <td id="T_f92a8_row108_col1" class="data row108 col1" >(1000, 512)</td>
      <td id="T_f92a8_row108_col2" class="data row108 col2" >512000</td>
      <td id="T_f92a8_row108_col3" class="data row108 col3" >512000</td>
      <td id="T_f92a8_row108_col4" class="data row108 col4" >(1000, 512)</td>
      <td id="T_f92a8_row108_col5" class="data row108 col5" >fc.weight</td>
    </tr>
    <tr>
      <th id="T_f92a8_level0_row109" class="row_heading level0 row109" >109</th>
      <td id="T_f92a8_row109_col0" class="data row109 col0" >output_layers.2.bias</td>
      <td id="T_f92a8_row109_col1" class="data row109 col1" >(1000,)</td>
      <td id="T_f92a8_row109_col2" class="data row109 col2" >1000</td>
      <td id="T_f92a8_row109_col3" class="data row109 col3" >1000</td>
      <td id="T_f92a8_row109_col4" class="data row109 col4" >(1000,)</td>
      <td id="T_f92a8_row109_col5" class="data row109 col5" >fc.bias</td>
    </tr>
  </tbody>
</table>




```python
pretrained_resnet50 = t.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
my_resnet50 = copy_weights(my_resnet50, pretrained_resnet50)
print_param_count(my_resnet50, pretrained_resnet50)
```

    Using cache found in /root/.cache/torch/hub/pytorch_vision_v0.10.0
    /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.
      warnings.warn(msg)


    Model 1, total params = 25557032
    Model 2, total params = 25557032
    All parameter counts match!



<style type="text/css">
#T_91102_row0_col2, #T_91102_row0_col3 {
  background-color: #238a8d;
  color: #f1f1f1;
}
#T_91102_row1_col2, #T_91102_row1_col3, #T_91102_row2_col2, #T_91102_row2_col3, #T_91102_row4_col2, #T_91102_row4_col3, #T_91102_row5_col2, #T_91102_row5_col3, #T_91102_row7_col2, #T_91102_row7_col3, #T_91102_row8_col2, #T_91102_row8_col3, #T_91102_row16_col2, #T_91102_row16_col3, #T_91102_row17_col2, #T_91102_row17_col3, #T_91102_row19_col2, #T_91102_row19_col3, #T_91102_row20_col2, #T_91102_row20_col3, #T_91102_row25_col2, #T_91102_row25_col3, #T_91102_row26_col2, #T_91102_row26_col3, #T_91102_row28_col2, #T_91102_row28_col3, #T_91102_row29_col2, #T_91102_row29_col3 {
  background-color: #440154;
  color: #f1f1f1;
}
#T_91102_row3_col2, #T_91102_row3_col3 {
  background-color: #2a778e;
  color: #f1f1f1;
}
#T_91102_row6_col2, #T_91102_row6_col3, #T_91102_row18_col2, #T_91102_row18_col3, #T_91102_row27_col2, #T_91102_row27_col3 {
  background-color: #23a983;
  color: #f1f1f1;
}
#T_91102_row9_col2, #T_91102_row9_col3, #T_91102_row12_col2, #T_91102_row12_col3, #T_91102_row15_col2, #T_91102_row15_col3, #T_91102_row21_col2, #T_91102_row21_col3, #T_91102_row24_col2, #T_91102_row24_col3, #T_91102_row30_col2, #T_91102_row30_col3 {
  background-color: #1f978b;
  color: #f1f1f1;
}
#T_91102_row10_col2, #T_91102_row10_col3, #T_91102_row11_col2, #T_91102_row11_col3, #T_91102_row13_col2, #T_91102_row13_col3, #T_91102_row14_col2, #T_91102_row14_col3, #T_91102_row22_col2, #T_91102_row22_col3, #T_91102_row23_col2, #T_91102_row23_col3, #T_91102_row31_col2, #T_91102_row31_col3, #T_91102_row32_col2, #T_91102_row32_col3, #T_91102_row73_col2, #T_91102_row73_col3, #T_91102_row74_col2, #T_91102_row74_col3, #T_91102_row76_col2, #T_91102_row76_col3, #T_91102_row77_col2, #T_91102_row77_col3, #T_91102_row85_col2, #T_91102_row85_col3, #T_91102_row86_col2, #T_91102_row86_col3, #T_91102_row88_col2, #T_91102_row88_col3, #T_91102_row89_col2, #T_91102_row89_col3, #T_91102_row94_col2, #T_91102_row94_col3, #T_91102_row95_col2, #T_91102_row95_col3, #T_91102_row97_col2, #T_91102_row97_col3, #T_91102_row98_col2, #T_91102_row98_col3, #T_91102_row103_col2, #T_91102_row103_col3, #T_91102_row104_col2, #T_91102_row104_col3, #T_91102_row106_col2, #T_91102_row106_col3, #T_91102_row107_col2, #T_91102_row107_col3, #T_91102_row112_col2, #T_91102_row112_col3, #T_91102_row113_col2, #T_91102_row113_col3, #T_91102_row115_col2, #T_91102_row115_col3, #T_91102_row116_col2, #T_91102_row116_col3, #T_91102_row121_col2, #T_91102_row121_col3, #T_91102_row122_col2, #T_91102_row122_col3, #T_91102_row124_col2, #T_91102_row124_col3, #T_91102_row125_col2, #T_91102_row125_col3 {
  background-color: #472e7c;
  color: #f1f1f1;
}
#T_91102_row33_col2, #T_91102_row33_col3 {
  background-color: #21a685;
  color: #f1f1f1;
}
#T_91102_row34_col2, #T_91102_row34_col3, #T_91102_row35_col2, #T_91102_row35_col3, #T_91102_row37_col2, #T_91102_row37_col3, #T_91102_row38_col2, #T_91102_row38_col3, #T_91102_row46_col2, #T_91102_row46_col3, #T_91102_row47_col2, #T_91102_row47_col3, #T_91102_row49_col2, #T_91102_row49_col3, #T_91102_row50_col2, #T_91102_row50_col3, #T_91102_row55_col2, #T_91102_row55_col3, #T_91102_row56_col2, #T_91102_row56_col3, #T_91102_row58_col2, #T_91102_row58_col3, #T_91102_row59_col2, #T_91102_row59_col3, #T_91102_row64_col2, #T_91102_row64_col3, #T_91102_row65_col2, #T_91102_row65_col3, #T_91102_row67_col2, #T_91102_row67_col3, #T_91102_row68_col2, #T_91102_row68_col3 {
  background-color: #48186a;
  color: #f1f1f1;
}
#T_91102_row36_col2, #T_91102_row36_col3, #T_91102_row48_col2, #T_91102_row48_col3, #T_91102_row57_col2, #T_91102_row57_col3, #T_91102_row66_col2, #T_91102_row66_col3 {
  background-color: #56c667;
  color: #000000;
}
#T_91102_row39_col2, #T_91102_row39_col3, #T_91102_row45_col2, #T_91102_row45_col3, #T_91102_row51_col2, #T_91102_row51_col3, #T_91102_row54_col2, #T_91102_row54_col3, #T_91102_row60_col2, #T_91102_row60_col3, #T_91102_row63_col2, #T_91102_row63_col3, #T_91102_row69_col2, #T_91102_row69_col3 {
  background-color: #32b67a;
  color: #f1f1f1;
}
#T_91102_row40_col2, #T_91102_row40_col3, #T_91102_row41_col2, #T_91102_row41_col3, #T_91102_row43_col2, #T_91102_row43_col3, #T_91102_row44_col2, #T_91102_row44_col3, #T_91102_row52_col2, #T_91102_row52_col3, #T_91102_row53_col2, #T_91102_row53_col3, #T_91102_row61_col2, #T_91102_row61_col3, #T_91102_row62_col2, #T_91102_row62_col3, #T_91102_row70_col2, #T_91102_row70_col3, #T_91102_row71_col2, #T_91102_row71_col3, #T_91102_row130_col2, #T_91102_row130_col3, #T_91102_row131_col2, #T_91102_row131_col3, #T_91102_row133_col2, #T_91102_row133_col3, #T_91102_row134_col2, #T_91102_row134_col3, #T_91102_row142_col2, #T_91102_row142_col3, #T_91102_row143_col2, #T_91102_row143_col3, #T_91102_row145_col2, #T_91102_row145_col3, #T_91102_row146_col2, #T_91102_row146_col3, #T_91102_row151_col2, #T_91102_row151_col3, #T_91102_row152_col2, #T_91102_row152_col3, #T_91102_row154_col2, #T_91102_row154_col3, #T_91102_row155_col2, #T_91102_row155_col3 {
  background-color: #414287;
  color: #f1f1f1;
}
#T_91102_row42_col2, #T_91102_row42_col3, #T_91102_row72_col2, #T_91102_row72_col3 {
  background-color: #50c46a;
  color: #000000;
}
#T_91102_row75_col2, #T_91102_row75_col3, #T_91102_row87_col2, #T_91102_row87_col3, #T_91102_row96_col2, #T_91102_row96_col3, #T_91102_row105_col2, #T_91102_row105_col3, #T_91102_row114_col2, #T_91102_row114_col3, #T_91102_row123_col2, #T_91102_row123_col3 {
  background-color: #a8db34;
  color: #000000;
}
#T_91102_row78_col2, #T_91102_row78_col3, #T_91102_row84_col2, #T_91102_row84_col3, #T_91102_row90_col2, #T_91102_row90_col3, #T_91102_row93_col2, #T_91102_row93_col3, #T_91102_row99_col2, #T_91102_row99_col3, #T_91102_row102_col2, #T_91102_row102_col3, #T_91102_row108_col2, #T_91102_row108_col3, #T_91102_row111_col2, #T_91102_row111_col3, #T_91102_row117_col2, #T_91102_row117_col3, #T_91102_row120_col2, #T_91102_row120_col3, #T_91102_row126_col2, #T_91102_row126_col3 {
  background-color: #75d054;
  color: #000000;
}
#T_91102_row79_col2, #T_91102_row79_col3, #T_91102_row80_col2, #T_91102_row80_col3, #T_91102_row82_col2, #T_91102_row82_col3, #T_91102_row83_col2, #T_91102_row83_col3, #T_91102_row91_col2, #T_91102_row91_col3, #T_91102_row92_col2, #T_91102_row92_col3, #T_91102_row100_col2, #T_91102_row100_col3, #T_91102_row101_col2, #T_91102_row101_col3, #T_91102_row109_col2, #T_91102_row109_col3, #T_91102_row110_col2, #T_91102_row110_col3, #T_91102_row118_col2, #T_91102_row118_col3, #T_91102_row119_col2, #T_91102_row119_col3, #T_91102_row127_col2, #T_91102_row127_col3, #T_91102_row128_col2, #T_91102_row128_col3 {
  background-color: #39558c;
  color: #f1f1f1;
}
#T_91102_row81_col2, #T_91102_row81_col3, #T_91102_row129_col2, #T_91102_row129_col3 {
  background-color: #a0da39;
  color: #000000;
}
#T_91102_row132_col2, #T_91102_row132_col3, #T_91102_row144_col2, #T_91102_row144_col3, #T_91102_row153_col2, #T_91102_row153_col3 {
  background-color: #fde725;
  color: #000000;
}
#T_91102_row135_col2, #T_91102_row135_col3, #T_91102_row141_col2, #T_91102_row141_col3, #T_91102_row147_col2, #T_91102_row147_col3, #T_91102_row150_col2, #T_91102_row150_col3, #T_91102_row156_col2, #T_91102_row156_col3 {
  background-color: #cde11d;
  color: #000000;
}
#T_91102_row136_col2, #T_91102_row136_col3, #T_91102_row137_col2, #T_91102_row137_col3, #T_91102_row139_col2, #T_91102_row139_col3, #T_91102_row140_col2, #T_91102_row140_col3, #T_91102_row148_col2, #T_91102_row148_col3, #T_91102_row149_col2, #T_91102_row149_col3, #T_91102_row157_col2, #T_91102_row157_col3, #T_91102_row158_col2, #T_91102_row158_col3 {
  background-color: #31678e;
  color: #f1f1f1;
}
#T_91102_row138_col2, #T_91102_row138_col3 {
  background-color: #f8e621;
  color: #000000;
}
#T_91102_row159_col2, #T_91102_row159_col3 {
  background-color: #f6e620;
  color: #000000;
}
#T_91102_row160_col2, #T_91102_row160_col3 {
  background-color: #3a548c;
  color: #f1f1f1;
}
</style>
<table id="T_91102" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_91102_level0_col0" class="col_heading level0 col0" >name_1</th>
      <th id="T_91102_level0_col1" class="col_heading level0 col1" >shape_1</th>
      <th id="T_91102_level0_col2" class="col_heading level0 col2" >num_params_1</th>
      <th id="T_91102_level0_col3" class="col_heading level0 col3" >num_params_2</th>
      <th id="T_91102_level0_col4" class="col_heading level0 col4" >shape_2</th>
      <th id="T_91102_level0_col5" class="col_heading level0 col5" >name_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_91102_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_91102_row0_col0" class="data row0 col0" >input_layers.0.conv.weight</td>
      <td id="T_91102_row0_col1" class="data row0 col1" >(64, 3, 7, 7)</td>
      <td id="T_91102_row0_col2" class="data row0 col2" >9408</td>
      <td id="T_91102_row0_col3" class="data row0 col3" >9408</td>
      <td id="T_91102_row0_col4" class="data row0 col4" >(64, 3, 7, 7)</td>
      <td id="T_91102_row0_col5" class="data row0 col5" >conv1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_91102_row1_col0" class="data row1 col0" >input_layers.0.batchnorm2d.weight</td>
      <td id="T_91102_row1_col1" class="data row1 col1" >(64,)</td>
      <td id="T_91102_row1_col2" class="data row1 col2" >64</td>
      <td id="T_91102_row1_col3" class="data row1 col3" >64</td>
      <td id="T_91102_row1_col4" class="data row1 col4" >(64,)</td>
      <td id="T_91102_row1_col5" class="data row1 col5" >bn1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_91102_row2_col0" class="data row2 col0" >input_layers.0.batchnorm2d.bias</td>
      <td id="T_91102_row2_col1" class="data row2 col1" >(64,)</td>
      <td id="T_91102_row2_col2" class="data row2 col2" >64</td>
      <td id="T_91102_row2_col3" class="data row2 col3" >64</td>
      <td id="T_91102_row2_col4" class="data row2 col4" >(64,)</td>
      <td id="T_91102_row2_col5" class="data row2 col5" >bn1.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_91102_row3_col0" class="data row3 col0" >block_groups.0.block_group.0.left.conv_block.0.conv.weight</td>
      <td id="T_91102_row3_col1" class="data row3 col1" >(64, 64, 1, 1)</td>
      <td id="T_91102_row3_col2" class="data row3 col2" >4096</td>
      <td id="T_91102_row3_col3" class="data row3 col3" >4096</td>
      <td id="T_91102_row3_col4" class="data row3 col4" >(64, 64, 1, 1)</td>
      <td id="T_91102_row3_col5" class="data row3 col5" >layer1.0.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_91102_row4_col0" class="data row4 col0" >block_groups.0.block_group.0.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_91102_row4_col1" class="data row4 col1" >(64,)</td>
      <td id="T_91102_row4_col2" class="data row4 col2" >64</td>
      <td id="T_91102_row4_col3" class="data row4 col3" >64</td>
      <td id="T_91102_row4_col4" class="data row4 col4" >(64,)</td>
      <td id="T_91102_row4_col5" class="data row4 col5" >layer1.0.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_91102_row5_col0" class="data row5 col0" >block_groups.0.block_group.0.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_91102_row5_col1" class="data row5 col1" >(64,)</td>
      <td id="T_91102_row5_col2" class="data row5 col2" >64</td>
      <td id="T_91102_row5_col3" class="data row5 col3" >64</td>
      <td id="T_91102_row5_col4" class="data row5 col4" >(64,)</td>
      <td id="T_91102_row5_col5" class="data row5 col5" >layer1.0.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_91102_row6_col0" class="data row6 col0" >block_groups.0.block_group.0.left.conv_block.1.conv.weight</td>
      <td id="T_91102_row6_col1" class="data row6 col1" >(64, 64, 3, 3)</td>
      <td id="T_91102_row6_col2" class="data row6 col2" >36864</td>
      <td id="T_91102_row6_col3" class="data row6 col3" >36864</td>
      <td id="T_91102_row6_col4" class="data row6 col4" >(64, 64, 3, 3)</td>
      <td id="T_91102_row6_col5" class="data row6 col5" >layer1.0.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_91102_row7_col0" class="data row7 col0" >block_groups.0.block_group.0.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_91102_row7_col1" class="data row7 col1" >(64,)</td>
      <td id="T_91102_row7_col2" class="data row7 col2" >64</td>
      <td id="T_91102_row7_col3" class="data row7 col3" >64</td>
      <td id="T_91102_row7_col4" class="data row7 col4" >(64,)</td>
      <td id="T_91102_row7_col5" class="data row7 col5" >layer1.0.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_91102_row8_col0" class="data row8 col0" >block_groups.0.block_group.0.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_91102_row8_col1" class="data row8 col1" >(64,)</td>
      <td id="T_91102_row8_col2" class="data row8 col2" >64</td>
      <td id="T_91102_row8_col3" class="data row8 col3" >64</td>
      <td id="T_91102_row8_col4" class="data row8 col4" >(64,)</td>
      <td id="T_91102_row8_col5" class="data row8 col5" >layer1.0.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_91102_row9_col0" class="data row9 col0" >block_groups.0.block_group.0.left.conv_block.2.conv.weight</td>
      <td id="T_91102_row9_col1" class="data row9 col1" >(256, 64, 1, 1)</td>
      <td id="T_91102_row9_col2" class="data row9 col2" >16384</td>
      <td id="T_91102_row9_col3" class="data row9 col3" >16384</td>
      <td id="T_91102_row9_col4" class="data row9 col4" >(256, 64, 1, 1)</td>
      <td id="T_91102_row9_col5" class="data row9 col5" >layer1.0.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_91102_row10_col0" class="data row10 col0" >block_groups.0.block_group.0.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_91102_row10_col1" class="data row10 col1" >(256,)</td>
      <td id="T_91102_row10_col2" class="data row10 col2" >256</td>
      <td id="T_91102_row10_col3" class="data row10 col3" >256</td>
      <td id="T_91102_row10_col4" class="data row10 col4" >(256,)</td>
      <td id="T_91102_row10_col5" class="data row10 col5" >layer1.0.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_91102_row11_col0" class="data row11 col0" >block_groups.0.block_group.0.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_91102_row11_col1" class="data row11 col1" >(256,)</td>
      <td id="T_91102_row11_col2" class="data row11 col2" >256</td>
      <td id="T_91102_row11_col3" class="data row11 col3" >256</td>
      <td id="T_91102_row11_col4" class="data row11 col4" >(256,)</td>
      <td id="T_91102_row11_col5" class="data row11 col5" >layer1.0.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_91102_row12_col0" class="data row12 col0" >block_groups.0.block_group.0.right.conv.weight</td>
      <td id="T_91102_row12_col1" class="data row12 col1" >(256, 64, 1, 1)</td>
      <td id="T_91102_row12_col2" class="data row12 col2" >16384</td>
      <td id="T_91102_row12_col3" class="data row12 col3" >16384</td>
      <td id="T_91102_row12_col4" class="data row12 col4" >(256, 64, 1, 1)</td>
      <td id="T_91102_row12_col5" class="data row12 col5" >layer1.0.downsample.0.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_91102_row13_col0" class="data row13 col0" >block_groups.0.block_group.0.right.batchnorm2d.weight</td>
      <td id="T_91102_row13_col1" class="data row13 col1" >(256,)</td>
      <td id="T_91102_row13_col2" class="data row13 col2" >256</td>
      <td id="T_91102_row13_col3" class="data row13 col3" >256</td>
      <td id="T_91102_row13_col4" class="data row13 col4" >(256,)</td>
      <td id="T_91102_row13_col5" class="data row13 col5" >layer1.0.downsample.1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_91102_row14_col0" class="data row14 col0" >block_groups.0.block_group.0.right.batchnorm2d.bias</td>
      <td id="T_91102_row14_col1" class="data row14 col1" >(256,)</td>
      <td id="T_91102_row14_col2" class="data row14 col2" >256</td>
      <td id="T_91102_row14_col3" class="data row14 col3" >256</td>
      <td id="T_91102_row14_col4" class="data row14 col4" >(256,)</td>
      <td id="T_91102_row14_col5" class="data row14 col5" >layer1.0.downsample.1.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_91102_row15_col0" class="data row15 col0" >block_groups.0.block_group.1.left.conv_block.0.conv.weight</td>
      <td id="T_91102_row15_col1" class="data row15 col1" >(64, 256, 1, 1)</td>
      <td id="T_91102_row15_col2" class="data row15 col2" >16384</td>
      <td id="T_91102_row15_col3" class="data row15 col3" >16384</td>
      <td id="T_91102_row15_col4" class="data row15 col4" >(64, 256, 1, 1)</td>
      <td id="T_91102_row15_col5" class="data row15 col5" >layer1.1.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_91102_row16_col0" class="data row16 col0" >block_groups.0.block_group.1.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_91102_row16_col1" class="data row16 col1" >(64,)</td>
      <td id="T_91102_row16_col2" class="data row16 col2" >64</td>
      <td id="T_91102_row16_col3" class="data row16 col3" >64</td>
      <td id="T_91102_row16_col4" class="data row16 col4" >(64,)</td>
      <td id="T_91102_row16_col5" class="data row16 col5" >layer1.1.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_91102_row17_col0" class="data row17 col0" >block_groups.0.block_group.1.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_91102_row17_col1" class="data row17 col1" >(64,)</td>
      <td id="T_91102_row17_col2" class="data row17 col2" >64</td>
      <td id="T_91102_row17_col3" class="data row17 col3" >64</td>
      <td id="T_91102_row17_col4" class="data row17 col4" >(64,)</td>
      <td id="T_91102_row17_col5" class="data row17 col5" >layer1.1.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_91102_row18_col0" class="data row18 col0" >block_groups.0.block_group.1.left.conv_block.1.conv.weight</td>
      <td id="T_91102_row18_col1" class="data row18 col1" >(64, 64, 3, 3)</td>
      <td id="T_91102_row18_col2" class="data row18 col2" >36864</td>
      <td id="T_91102_row18_col3" class="data row18 col3" >36864</td>
      <td id="T_91102_row18_col4" class="data row18 col4" >(64, 64, 3, 3)</td>
      <td id="T_91102_row18_col5" class="data row18 col5" >layer1.1.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_91102_row19_col0" class="data row19 col0" >block_groups.0.block_group.1.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_91102_row19_col1" class="data row19 col1" >(64,)</td>
      <td id="T_91102_row19_col2" class="data row19 col2" >64</td>
      <td id="T_91102_row19_col3" class="data row19 col3" >64</td>
      <td id="T_91102_row19_col4" class="data row19 col4" >(64,)</td>
      <td id="T_91102_row19_col5" class="data row19 col5" >layer1.1.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row20" class="row_heading level0 row20" >20</th>
      <td id="T_91102_row20_col0" class="data row20 col0" >block_groups.0.block_group.1.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_91102_row20_col1" class="data row20 col1" >(64,)</td>
      <td id="T_91102_row20_col2" class="data row20 col2" >64</td>
      <td id="T_91102_row20_col3" class="data row20 col3" >64</td>
      <td id="T_91102_row20_col4" class="data row20 col4" >(64,)</td>
      <td id="T_91102_row20_col5" class="data row20 col5" >layer1.1.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row21" class="row_heading level0 row21" >21</th>
      <td id="T_91102_row21_col0" class="data row21 col0" >block_groups.0.block_group.1.left.conv_block.2.conv.weight</td>
      <td id="T_91102_row21_col1" class="data row21 col1" >(256, 64, 1, 1)</td>
      <td id="T_91102_row21_col2" class="data row21 col2" >16384</td>
      <td id="T_91102_row21_col3" class="data row21 col3" >16384</td>
      <td id="T_91102_row21_col4" class="data row21 col4" >(256, 64, 1, 1)</td>
      <td id="T_91102_row21_col5" class="data row21 col5" >layer1.1.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row22" class="row_heading level0 row22" >22</th>
      <td id="T_91102_row22_col0" class="data row22 col0" >block_groups.0.block_group.1.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_91102_row22_col1" class="data row22 col1" >(256,)</td>
      <td id="T_91102_row22_col2" class="data row22 col2" >256</td>
      <td id="T_91102_row22_col3" class="data row22 col3" >256</td>
      <td id="T_91102_row22_col4" class="data row22 col4" >(256,)</td>
      <td id="T_91102_row22_col5" class="data row22 col5" >layer1.1.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row23" class="row_heading level0 row23" >23</th>
      <td id="T_91102_row23_col0" class="data row23 col0" >block_groups.0.block_group.1.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_91102_row23_col1" class="data row23 col1" >(256,)</td>
      <td id="T_91102_row23_col2" class="data row23 col2" >256</td>
      <td id="T_91102_row23_col3" class="data row23 col3" >256</td>
      <td id="T_91102_row23_col4" class="data row23 col4" >(256,)</td>
      <td id="T_91102_row23_col5" class="data row23 col5" >layer1.1.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row24" class="row_heading level0 row24" >24</th>
      <td id="T_91102_row24_col0" class="data row24 col0" >block_groups.0.block_group.2.left.conv_block.0.conv.weight</td>
      <td id="T_91102_row24_col1" class="data row24 col1" >(64, 256, 1, 1)</td>
      <td id="T_91102_row24_col2" class="data row24 col2" >16384</td>
      <td id="T_91102_row24_col3" class="data row24 col3" >16384</td>
      <td id="T_91102_row24_col4" class="data row24 col4" >(64, 256, 1, 1)</td>
      <td id="T_91102_row24_col5" class="data row24 col5" >layer1.2.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row25" class="row_heading level0 row25" >25</th>
      <td id="T_91102_row25_col0" class="data row25 col0" >block_groups.0.block_group.2.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_91102_row25_col1" class="data row25 col1" >(64,)</td>
      <td id="T_91102_row25_col2" class="data row25 col2" >64</td>
      <td id="T_91102_row25_col3" class="data row25 col3" >64</td>
      <td id="T_91102_row25_col4" class="data row25 col4" >(64,)</td>
      <td id="T_91102_row25_col5" class="data row25 col5" >layer1.2.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row26" class="row_heading level0 row26" >26</th>
      <td id="T_91102_row26_col0" class="data row26 col0" >block_groups.0.block_group.2.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_91102_row26_col1" class="data row26 col1" >(64,)</td>
      <td id="T_91102_row26_col2" class="data row26 col2" >64</td>
      <td id="T_91102_row26_col3" class="data row26 col3" >64</td>
      <td id="T_91102_row26_col4" class="data row26 col4" >(64,)</td>
      <td id="T_91102_row26_col5" class="data row26 col5" >layer1.2.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row27" class="row_heading level0 row27" >27</th>
      <td id="T_91102_row27_col0" class="data row27 col0" >block_groups.0.block_group.2.left.conv_block.1.conv.weight</td>
      <td id="T_91102_row27_col1" class="data row27 col1" >(64, 64, 3, 3)</td>
      <td id="T_91102_row27_col2" class="data row27 col2" >36864</td>
      <td id="T_91102_row27_col3" class="data row27 col3" >36864</td>
      <td id="T_91102_row27_col4" class="data row27 col4" >(64, 64, 3, 3)</td>
      <td id="T_91102_row27_col5" class="data row27 col5" >layer1.2.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row28" class="row_heading level0 row28" >28</th>
      <td id="T_91102_row28_col0" class="data row28 col0" >block_groups.0.block_group.2.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_91102_row28_col1" class="data row28 col1" >(64,)</td>
      <td id="T_91102_row28_col2" class="data row28 col2" >64</td>
      <td id="T_91102_row28_col3" class="data row28 col3" >64</td>
      <td id="T_91102_row28_col4" class="data row28 col4" >(64,)</td>
      <td id="T_91102_row28_col5" class="data row28 col5" >layer1.2.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row29" class="row_heading level0 row29" >29</th>
      <td id="T_91102_row29_col0" class="data row29 col0" >block_groups.0.block_group.2.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_91102_row29_col1" class="data row29 col1" >(64,)</td>
      <td id="T_91102_row29_col2" class="data row29 col2" >64</td>
      <td id="T_91102_row29_col3" class="data row29 col3" >64</td>
      <td id="T_91102_row29_col4" class="data row29 col4" >(64,)</td>
      <td id="T_91102_row29_col5" class="data row29 col5" >layer1.2.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row30" class="row_heading level0 row30" >30</th>
      <td id="T_91102_row30_col0" class="data row30 col0" >block_groups.0.block_group.2.left.conv_block.2.conv.weight</td>
      <td id="T_91102_row30_col1" class="data row30 col1" >(256, 64, 1, 1)</td>
      <td id="T_91102_row30_col2" class="data row30 col2" >16384</td>
      <td id="T_91102_row30_col3" class="data row30 col3" >16384</td>
      <td id="T_91102_row30_col4" class="data row30 col4" >(256, 64, 1, 1)</td>
      <td id="T_91102_row30_col5" class="data row30 col5" >layer1.2.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row31" class="row_heading level0 row31" >31</th>
      <td id="T_91102_row31_col0" class="data row31 col0" >block_groups.0.block_group.2.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_91102_row31_col1" class="data row31 col1" >(256,)</td>
      <td id="T_91102_row31_col2" class="data row31 col2" >256</td>
      <td id="T_91102_row31_col3" class="data row31 col3" >256</td>
      <td id="T_91102_row31_col4" class="data row31 col4" >(256,)</td>
      <td id="T_91102_row31_col5" class="data row31 col5" >layer1.2.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row32" class="row_heading level0 row32" >32</th>
      <td id="T_91102_row32_col0" class="data row32 col0" >block_groups.0.block_group.2.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_91102_row32_col1" class="data row32 col1" >(256,)</td>
      <td id="T_91102_row32_col2" class="data row32 col2" >256</td>
      <td id="T_91102_row32_col3" class="data row32 col3" >256</td>
      <td id="T_91102_row32_col4" class="data row32 col4" >(256,)</td>
      <td id="T_91102_row32_col5" class="data row32 col5" >layer1.2.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row33" class="row_heading level0 row33" >33</th>
      <td id="T_91102_row33_col0" class="data row33 col0" >block_groups.1.block_group.0.left.conv_block.0.conv.weight</td>
      <td id="T_91102_row33_col1" class="data row33 col1" >(128, 256, 1, 1)</td>
      <td id="T_91102_row33_col2" class="data row33 col2" >32768</td>
      <td id="T_91102_row33_col3" class="data row33 col3" >32768</td>
      <td id="T_91102_row33_col4" class="data row33 col4" >(128, 256, 1, 1)</td>
      <td id="T_91102_row33_col5" class="data row33 col5" >layer2.0.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row34" class="row_heading level0 row34" >34</th>
      <td id="T_91102_row34_col0" class="data row34 col0" >block_groups.1.block_group.0.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_91102_row34_col1" class="data row34 col1" >(128,)</td>
      <td id="T_91102_row34_col2" class="data row34 col2" >128</td>
      <td id="T_91102_row34_col3" class="data row34 col3" >128</td>
      <td id="T_91102_row34_col4" class="data row34 col4" >(128,)</td>
      <td id="T_91102_row34_col5" class="data row34 col5" >layer2.0.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row35" class="row_heading level0 row35" >35</th>
      <td id="T_91102_row35_col0" class="data row35 col0" >block_groups.1.block_group.0.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_91102_row35_col1" class="data row35 col1" >(128,)</td>
      <td id="T_91102_row35_col2" class="data row35 col2" >128</td>
      <td id="T_91102_row35_col3" class="data row35 col3" >128</td>
      <td id="T_91102_row35_col4" class="data row35 col4" >(128,)</td>
      <td id="T_91102_row35_col5" class="data row35 col5" >layer2.0.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row36" class="row_heading level0 row36" >36</th>
      <td id="T_91102_row36_col0" class="data row36 col0" >block_groups.1.block_group.0.left.conv_block.1.conv.weight</td>
      <td id="T_91102_row36_col1" class="data row36 col1" >(128, 128, 3, 3)</td>
      <td id="T_91102_row36_col2" class="data row36 col2" >147456</td>
      <td id="T_91102_row36_col3" class="data row36 col3" >147456</td>
      <td id="T_91102_row36_col4" class="data row36 col4" >(128, 128, 3, 3)</td>
      <td id="T_91102_row36_col5" class="data row36 col5" >layer2.0.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row37" class="row_heading level0 row37" >37</th>
      <td id="T_91102_row37_col0" class="data row37 col0" >block_groups.1.block_group.0.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_91102_row37_col1" class="data row37 col1" >(128,)</td>
      <td id="T_91102_row37_col2" class="data row37 col2" >128</td>
      <td id="T_91102_row37_col3" class="data row37 col3" >128</td>
      <td id="T_91102_row37_col4" class="data row37 col4" >(128,)</td>
      <td id="T_91102_row37_col5" class="data row37 col5" >layer2.0.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row38" class="row_heading level0 row38" >38</th>
      <td id="T_91102_row38_col0" class="data row38 col0" >block_groups.1.block_group.0.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_91102_row38_col1" class="data row38 col1" >(128,)</td>
      <td id="T_91102_row38_col2" class="data row38 col2" >128</td>
      <td id="T_91102_row38_col3" class="data row38 col3" >128</td>
      <td id="T_91102_row38_col4" class="data row38 col4" >(128,)</td>
      <td id="T_91102_row38_col5" class="data row38 col5" >layer2.0.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row39" class="row_heading level0 row39" >39</th>
      <td id="T_91102_row39_col0" class="data row39 col0" >block_groups.1.block_group.0.left.conv_block.2.conv.weight</td>
      <td id="T_91102_row39_col1" class="data row39 col1" >(512, 128, 1, 1)</td>
      <td id="T_91102_row39_col2" class="data row39 col2" >65536</td>
      <td id="T_91102_row39_col3" class="data row39 col3" >65536</td>
      <td id="T_91102_row39_col4" class="data row39 col4" >(512, 128, 1, 1)</td>
      <td id="T_91102_row39_col5" class="data row39 col5" >layer2.0.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row40" class="row_heading level0 row40" >40</th>
      <td id="T_91102_row40_col0" class="data row40 col0" >block_groups.1.block_group.0.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_91102_row40_col1" class="data row40 col1" >(512,)</td>
      <td id="T_91102_row40_col2" class="data row40 col2" >512</td>
      <td id="T_91102_row40_col3" class="data row40 col3" >512</td>
      <td id="T_91102_row40_col4" class="data row40 col4" >(512,)</td>
      <td id="T_91102_row40_col5" class="data row40 col5" >layer2.0.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row41" class="row_heading level0 row41" >41</th>
      <td id="T_91102_row41_col0" class="data row41 col0" >block_groups.1.block_group.0.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_91102_row41_col1" class="data row41 col1" >(512,)</td>
      <td id="T_91102_row41_col2" class="data row41 col2" >512</td>
      <td id="T_91102_row41_col3" class="data row41 col3" >512</td>
      <td id="T_91102_row41_col4" class="data row41 col4" >(512,)</td>
      <td id="T_91102_row41_col5" class="data row41 col5" >layer2.0.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row42" class="row_heading level0 row42" >42</th>
      <td id="T_91102_row42_col0" class="data row42 col0" >block_groups.1.block_group.0.right.conv.weight</td>
      <td id="T_91102_row42_col1" class="data row42 col1" >(512, 256, 1, 1)</td>
      <td id="T_91102_row42_col2" class="data row42 col2" >131072</td>
      <td id="T_91102_row42_col3" class="data row42 col3" >131072</td>
      <td id="T_91102_row42_col4" class="data row42 col4" >(512, 256, 1, 1)</td>
      <td id="T_91102_row42_col5" class="data row42 col5" >layer2.0.downsample.0.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row43" class="row_heading level0 row43" >43</th>
      <td id="T_91102_row43_col0" class="data row43 col0" >block_groups.1.block_group.0.right.batchnorm2d.weight</td>
      <td id="T_91102_row43_col1" class="data row43 col1" >(512,)</td>
      <td id="T_91102_row43_col2" class="data row43 col2" >512</td>
      <td id="T_91102_row43_col3" class="data row43 col3" >512</td>
      <td id="T_91102_row43_col4" class="data row43 col4" >(512,)</td>
      <td id="T_91102_row43_col5" class="data row43 col5" >layer2.0.downsample.1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row44" class="row_heading level0 row44" >44</th>
      <td id="T_91102_row44_col0" class="data row44 col0" >block_groups.1.block_group.0.right.batchnorm2d.bias</td>
      <td id="T_91102_row44_col1" class="data row44 col1" >(512,)</td>
      <td id="T_91102_row44_col2" class="data row44 col2" >512</td>
      <td id="T_91102_row44_col3" class="data row44 col3" >512</td>
      <td id="T_91102_row44_col4" class="data row44 col4" >(512,)</td>
      <td id="T_91102_row44_col5" class="data row44 col5" >layer2.0.downsample.1.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row45" class="row_heading level0 row45" >45</th>
      <td id="T_91102_row45_col0" class="data row45 col0" >block_groups.1.block_group.1.left.conv_block.0.conv.weight</td>
      <td id="T_91102_row45_col1" class="data row45 col1" >(128, 512, 1, 1)</td>
      <td id="T_91102_row45_col2" class="data row45 col2" >65536</td>
      <td id="T_91102_row45_col3" class="data row45 col3" >65536</td>
      <td id="T_91102_row45_col4" class="data row45 col4" >(128, 512, 1, 1)</td>
      <td id="T_91102_row45_col5" class="data row45 col5" >layer2.1.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row46" class="row_heading level0 row46" >46</th>
      <td id="T_91102_row46_col0" class="data row46 col0" >block_groups.1.block_group.1.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_91102_row46_col1" class="data row46 col1" >(128,)</td>
      <td id="T_91102_row46_col2" class="data row46 col2" >128</td>
      <td id="T_91102_row46_col3" class="data row46 col3" >128</td>
      <td id="T_91102_row46_col4" class="data row46 col4" >(128,)</td>
      <td id="T_91102_row46_col5" class="data row46 col5" >layer2.1.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row47" class="row_heading level0 row47" >47</th>
      <td id="T_91102_row47_col0" class="data row47 col0" >block_groups.1.block_group.1.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_91102_row47_col1" class="data row47 col1" >(128,)</td>
      <td id="T_91102_row47_col2" class="data row47 col2" >128</td>
      <td id="T_91102_row47_col3" class="data row47 col3" >128</td>
      <td id="T_91102_row47_col4" class="data row47 col4" >(128,)</td>
      <td id="T_91102_row47_col5" class="data row47 col5" >layer2.1.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row48" class="row_heading level0 row48" >48</th>
      <td id="T_91102_row48_col0" class="data row48 col0" >block_groups.1.block_group.1.left.conv_block.1.conv.weight</td>
      <td id="T_91102_row48_col1" class="data row48 col1" >(128, 128, 3, 3)</td>
      <td id="T_91102_row48_col2" class="data row48 col2" >147456</td>
      <td id="T_91102_row48_col3" class="data row48 col3" >147456</td>
      <td id="T_91102_row48_col4" class="data row48 col4" >(128, 128, 3, 3)</td>
      <td id="T_91102_row48_col5" class="data row48 col5" >layer2.1.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row49" class="row_heading level0 row49" >49</th>
      <td id="T_91102_row49_col0" class="data row49 col0" >block_groups.1.block_group.1.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_91102_row49_col1" class="data row49 col1" >(128,)</td>
      <td id="T_91102_row49_col2" class="data row49 col2" >128</td>
      <td id="T_91102_row49_col3" class="data row49 col3" >128</td>
      <td id="T_91102_row49_col4" class="data row49 col4" >(128,)</td>
      <td id="T_91102_row49_col5" class="data row49 col5" >layer2.1.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row50" class="row_heading level0 row50" >50</th>
      <td id="T_91102_row50_col0" class="data row50 col0" >block_groups.1.block_group.1.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_91102_row50_col1" class="data row50 col1" >(128,)</td>
      <td id="T_91102_row50_col2" class="data row50 col2" >128</td>
      <td id="T_91102_row50_col3" class="data row50 col3" >128</td>
      <td id="T_91102_row50_col4" class="data row50 col4" >(128,)</td>
      <td id="T_91102_row50_col5" class="data row50 col5" >layer2.1.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row51" class="row_heading level0 row51" >51</th>
      <td id="T_91102_row51_col0" class="data row51 col0" >block_groups.1.block_group.1.left.conv_block.2.conv.weight</td>
      <td id="T_91102_row51_col1" class="data row51 col1" >(512, 128, 1, 1)</td>
      <td id="T_91102_row51_col2" class="data row51 col2" >65536</td>
      <td id="T_91102_row51_col3" class="data row51 col3" >65536</td>
      <td id="T_91102_row51_col4" class="data row51 col4" >(512, 128, 1, 1)</td>
      <td id="T_91102_row51_col5" class="data row51 col5" >layer2.1.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row52" class="row_heading level0 row52" >52</th>
      <td id="T_91102_row52_col0" class="data row52 col0" >block_groups.1.block_group.1.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_91102_row52_col1" class="data row52 col1" >(512,)</td>
      <td id="T_91102_row52_col2" class="data row52 col2" >512</td>
      <td id="T_91102_row52_col3" class="data row52 col3" >512</td>
      <td id="T_91102_row52_col4" class="data row52 col4" >(512,)</td>
      <td id="T_91102_row52_col5" class="data row52 col5" >layer2.1.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row53" class="row_heading level0 row53" >53</th>
      <td id="T_91102_row53_col0" class="data row53 col0" >block_groups.1.block_group.1.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_91102_row53_col1" class="data row53 col1" >(512,)</td>
      <td id="T_91102_row53_col2" class="data row53 col2" >512</td>
      <td id="T_91102_row53_col3" class="data row53 col3" >512</td>
      <td id="T_91102_row53_col4" class="data row53 col4" >(512,)</td>
      <td id="T_91102_row53_col5" class="data row53 col5" >layer2.1.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row54" class="row_heading level0 row54" >54</th>
      <td id="T_91102_row54_col0" class="data row54 col0" >block_groups.1.block_group.2.left.conv_block.0.conv.weight</td>
      <td id="T_91102_row54_col1" class="data row54 col1" >(128, 512, 1, 1)</td>
      <td id="T_91102_row54_col2" class="data row54 col2" >65536</td>
      <td id="T_91102_row54_col3" class="data row54 col3" >65536</td>
      <td id="T_91102_row54_col4" class="data row54 col4" >(128, 512, 1, 1)</td>
      <td id="T_91102_row54_col5" class="data row54 col5" >layer2.2.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row55" class="row_heading level0 row55" >55</th>
      <td id="T_91102_row55_col0" class="data row55 col0" >block_groups.1.block_group.2.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_91102_row55_col1" class="data row55 col1" >(128,)</td>
      <td id="T_91102_row55_col2" class="data row55 col2" >128</td>
      <td id="T_91102_row55_col3" class="data row55 col3" >128</td>
      <td id="T_91102_row55_col4" class="data row55 col4" >(128,)</td>
      <td id="T_91102_row55_col5" class="data row55 col5" >layer2.2.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row56" class="row_heading level0 row56" >56</th>
      <td id="T_91102_row56_col0" class="data row56 col0" >block_groups.1.block_group.2.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_91102_row56_col1" class="data row56 col1" >(128,)</td>
      <td id="T_91102_row56_col2" class="data row56 col2" >128</td>
      <td id="T_91102_row56_col3" class="data row56 col3" >128</td>
      <td id="T_91102_row56_col4" class="data row56 col4" >(128,)</td>
      <td id="T_91102_row56_col5" class="data row56 col5" >layer2.2.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row57" class="row_heading level0 row57" >57</th>
      <td id="T_91102_row57_col0" class="data row57 col0" >block_groups.1.block_group.2.left.conv_block.1.conv.weight</td>
      <td id="T_91102_row57_col1" class="data row57 col1" >(128, 128, 3, 3)</td>
      <td id="T_91102_row57_col2" class="data row57 col2" >147456</td>
      <td id="T_91102_row57_col3" class="data row57 col3" >147456</td>
      <td id="T_91102_row57_col4" class="data row57 col4" >(128, 128, 3, 3)</td>
      <td id="T_91102_row57_col5" class="data row57 col5" >layer2.2.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row58" class="row_heading level0 row58" >58</th>
      <td id="T_91102_row58_col0" class="data row58 col0" >block_groups.1.block_group.2.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_91102_row58_col1" class="data row58 col1" >(128,)</td>
      <td id="T_91102_row58_col2" class="data row58 col2" >128</td>
      <td id="T_91102_row58_col3" class="data row58 col3" >128</td>
      <td id="T_91102_row58_col4" class="data row58 col4" >(128,)</td>
      <td id="T_91102_row58_col5" class="data row58 col5" >layer2.2.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row59" class="row_heading level0 row59" >59</th>
      <td id="T_91102_row59_col0" class="data row59 col0" >block_groups.1.block_group.2.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_91102_row59_col1" class="data row59 col1" >(128,)</td>
      <td id="T_91102_row59_col2" class="data row59 col2" >128</td>
      <td id="T_91102_row59_col3" class="data row59 col3" >128</td>
      <td id="T_91102_row59_col4" class="data row59 col4" >(128,)</td>
      <td id="T_91102_row59_col5" class="data row59 col5" >layer2.2.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row60" class="row_heading level0 row60" >60</th>
      <td id="T_91102_row60_col0" class="data row60 col0" >block_groups.1.block_group.2.left.conv_block.2.conv.weight</td>
      <td id="T_91102_row60_col1" class="data row60 col1" >(512, 128, 1, 1)</td>
      <td id="T_91102_row60_col2" class="data row60 col2" >65536</td>
      <td id="T_91102_row60_col3" class="data row60 col3" >65536</td>
      <td id="T_91102_row60_col4" class="data row60 col4" >(512, 128, 1, 1)</td>
      <td id="T_91102_row60_col5" class="data row60 col5" >layer2.2.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row61" class="row_heading level0 row61" >61</th>
      <td id="T_91102_row61_col0" class="data row61 col0" >block_groups.1.block_group.2.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_91102_row61_col1" class="data row61 col1" >(512,)</td>
      <td id="T_91102_row61_col2" class="data row61 col2" >512</td>
      <td id="T_91102_row61_col3" class="data row61 col3" >512</td>
      <td id="T_91102_row61_col4" class="data row61 col4" >(512,)</td>
      <td id="T_91102_row61_col5" class="data row61 col5" >layer2.2.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row62" class="row_heading level0 row62" >62</th>
      <td id="T_91102_row62_col0" class="data row62 col0" >block_groups.1.block_group.2.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_91102_row62_col1" class="data row62 col1" >(512,)</td>
      <td id="T_91102_row62_col2" class="data row62 col2" >512</td>
      <td id="T_91102_row62_col3" class="data row62 col3" >512</td>
      <td id="T_91102_row62_col4" class="data row62 col4" >(512,)</td>
      <td id="T_91102_row62_col5" class="data row62 col5" >layer2.2.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row63" class="row_heading level0 row63" >63</th>
      <td id="T_91102_row63_col0" class="data row63 col0" >block_groups.1.block_group.3.left.conv_block.0.conv.weight</td>
      <td id="T_91102_row63_col1" class="data row63 col1" >(128, 512, 1, 1)</td>
      <td id="T_91102_row63_col2" class="data row63 col2" >65536</td>
      <td id="T_91102_row63_col3" class="data row63 col3" >65536</td>
      <td id="T_91102_row63_col4" class="data row63 col4" >(128, 512, 1, 1)</td>
      <td id="T_91102_row63_col5" class="data row63 col5" >layer2.3.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row64" class="row_heading level0 row64" >64</th>
      <td id="T_91102_row64_col0" class="data row64 col0" >block_groups.1.block_group.3.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_91102_row64_col1" class="data row64 col1" >(128,)</td>
      <td id="T_91102_row64_col2" class="data row64 col2" >128</td>
      <td id="T_91102_row64_col3" class="data row64 col3" >128</td>
      <td id="T_91102_row64_col4" class="data row64 col4" >(128,)</td>
      <td id="T_91102_row64_col5" class="data row64 col5" >layer2.3.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row65" class="row_heading level0 row65" >65</th>
      <td id="T_91102_row65_col0" class="data row65 col0" >block_groups.1.block_group.3.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_91102_row65_col1" class="data row65 col1" >(128,)</td>
      <td id="T_91102_row65_col2" class="data row65 col2" >128</td>
      <td id="T_91102_row65_col3" class="data row65 col3" >128</td>
      <td id="T_91102_row65_col4" class="data row65 col4" >(128,)</td>
      <td id="T_91102_row65_col5" class="data row65 col5" >layer2.3.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row66" class="row_heading level0 row66" >66</th>
      <td id="T_91102_row66_col0" class="data row66 col0" >block_groups.1.block_group.3.left.conv_block.1.conv.weight</td>
      <td id="T_91102_row66_col1" class="data row66 col1" >(128, 128, 3, 3)</td>
      <td id="T_91102_row66_col2" class="data row66 col2" >147456</td>
      <td id="T_91102_row66_col3" class="data row66 col3" >147456</td>
      <td id="T_91102_row66_col4" class="data row66 col4" >(128, 128, 3, 3)</td>
      <td id="T_91102_row66_col5" class="data row66 col5" >layer2.3.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row67" class="row_heading level0 row67" >67</th>
      <td id="T_91102_row67_col0" class="data row67 col0" >block_groups.1.block_group.3.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_91102_row67_col1" class="data row67 col1" >(128,)</td>
      <td id="T_91102_row67_col2" class="data row67 col2" >128</td>
      <td id="T_91102_row67_col3" class="data row67 col3" >128</td>
      <td id="T_91102_row67_col4" class="data row67 col4" >(128,)</td>
      <td id="T_91102_row67_col5" class="data row67 col5" >layer2.3.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row68" class="row_heading level0 row68" >68</th>
      <td id="T_91102_row68_col0" class="data row68 col0" >block_groups.1.block_group.3.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_91102_row68_col1" class="data row68 col1" >(128,)</td>
      <td id="T_91102_row68_col2" class="data row68 col2" >128</td>
      <td id="T_91102_row68_col3" class="data row68 col3" >128</td>
      <td id="T_91102_row68_col4" class="data row68 col4" >(128,)</td>
      <td id="T_91102_row68_col5" class="data row68 col5" >layer2.3.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row69" class="row_heading level0 row69" >69</th>
      <td id="T_91102_row69_col0" class="data row69 col0" >block_groups.1.block_group.3.left.conv_block.2.conv.weight</td>
      <td id="T_91102_row69_col1" class="data row69 col1" >(512, 128, 1, 1)</td>
      <td id="T_91102_row69_col2" class="data row69 col2" >65536</td>
      <td id="T_91102_row69_col3" class="data row69 col3" >65536</td>
      <td id="T_91102_row69_col4" class="data row69 col4" >(512, 128, 1, 1)</td>
      <td id="T_91102_row69_col5" class="data row69 col5" >layer2.3.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row70" class="row_heading level0 row70" >70</th>
      <td id="T_91102_row70_col0" class="data row70 col0" >block_groups.1.block_group.3.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_91102_row70_col1" class="data row70 col1" >(512,)</td>
      <td id="T_91102_row70_col2" class="data row70 col2" >512</td>
      <td id="T_91102_row70_col3" class="data row70 col3" >512</td>
      <td id="T_91102_row70_col4" class="data row70 col4" >(512,)</td>
      <td id="T_91102_row70_col5" class="data row70 col5" >layer2.3.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row71" class="row_heading level0 row71" >71</th>
      <td id="T_91102_row71_col0" class="data row71 col0" >block_groups.1.block_group.3.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_91102_row71_col1" class="data row71 col1" >(512,)</td>
      <td id="T_91102_row71_col2" class="data row71 col2" >512</td>
      <td id="T_91102_row71_col3" class="data row71 col3" >512</td>
      <td id="T_91102_row71_col4" class="data row71 col4" >(512,)</td>
      <td id="T_91102_row71_col5" class="data row71 col5" >layer2.3.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row72" class="row_heading level0 row72" >72</th>
      <td id="T_91102_row72_col0" class="data row72 col0" >block_groups.2.block_group.0.left.conv_block.0.conv.weight</td>
      <td id="T_91102_row72_col1" class="data row72 col1" >(256, 512, 1, 1)</td>
      <td id="T_91102_row72_col2" class="data row72 col2" >131072</td>
      <td id="T_91102_row72_col3" class="data row72 col3" >131072</td>
      <td id="T_91102_row72_col4" class="data row72 col4" >(256, 512, 1, 1)</td>
      <td id="T_91102_row72_col5" class="data row72 col5" >layer3.0.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row73" class="row_heading level0 row73" >73</th>
      <td id="T_91102_row73_col0" class="data row73 col0" >block_groups.2.block_group.0.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_91102_row73_col1" class="data row73 col1" >(256,)</td>
      <td id="T_91102_row73_col2" class="data row73 col2" >256</td>
      <td id="T_91102_row73_col3" class="data row73 col3" >256</td>
      <td id="T_91102_row73_col4" class="data row73 col4" >(256,)</td>
      <td id="T_91102_row73_col5" class="data row73 col5" >layer3.0.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row74" class="row_heading level0 row74" >74</th>
      <td id="T_91102_row74_col0" class="data row74 col0" >block_groups.2.block_group.0.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_91102_row74_col1" class="data row74 col1" >(256,)</td>
      <td id="T_91102_row74_col2" class="data row74 col2" >256</td>
      <td id="T_91102_row74_col3" class="data row74 col3" >256</td>
      <td id="T_91102_row74_col4" class="data row74 col4" >(256,)</td>
      <td id="T_91102_row74_col5" class="data row74 col5" >layer3.0.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row75" class="row_heading level0 row75" >75</th>
      <td id="T_91102_row75_col0" class="data row75 col0" >block_groups.2.block_group.0.left.conv_block.1.conv.weight</td>
      <td id="T_91102_row75_col1" class="data row75 col1" >(256, 256, 3, 3)</td>
      <td id="T_91102_row75_col2" class="data row75 col2" >589824</td>
      <td id="T_91102_row75_col3" class="data row75 col3" >589824</td>
      <td id="T_91102_row75_col4" class="data row75 col4" >(256, 256, 3, 3)</td>
      <td id="T_91102_row75_col5" class="data row75 col5" >layer3.0.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row76" class="row_heading level0 row76" >76</th>
      <td id="T_91102_row76_col0" class="data row76 col0" >block_groups.2.block_group.0.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_91102_row76_col1" class="data row76 col1" >(256,)</td>
      <td id="T_91102_row76_col2" class="data row76 col2" >256</td>
      <td id="T_91102_row76_col3" class="data row76 col3" >256</td>
      <td id="T_91102_row76_col4" class="data row76 col4" >(256,)</td>
      <td id="T_91102_row76_col5" class="data row76 col5" >layer3.0.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row77" class="row_heading level0 row77" >77</th>
      <td id="T_91102_row77_col0" class="data row77 col0" >block_groups.2.block_group.0.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_91102_row77_col1" class="data row77 col1" >(256,)</td>
      <td id="T_91102_row77_col2" class="data row77 col2" >256</td>
      <td id="T_91102_row77_col3" class="data row77 col3" >256</td>
      <td id="T_91102_row77_col4" class="data row77 col4" >(256,)</td>
      <td id="T_91102_row77_col5" class="data row77 col5" >layer3.0.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row78" class="row_heading level0 row78" >78</th>
      <td id="T_91102_row78_col0" class="data row78 col0" >block_groups.2.block_group.0.left.conv_block.2.conv.weight</td>
      <td id="T_91102_row78_col1" class="data row78 col1" >(1024, 256, 1, 1)</td>
      <td id="T_91102_row78_col2" class="data row78 col2" >262144</td>
      <td id="T_91102_row78_col3" class="data row78 col3" >262144</td>
      <td id="T_91102_row78_col4" class="data row78 col4" >(1024, 256, 1, 1)</td>
      <td id="T_91102_row78_col5" class="data row78 col5" >layer3.0.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row79" class="row_heading level0 row79" >79</th>
      <td id="T_91102_row79_col0" class="data row79 col0" >block_groups.2.block_group.0.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_91102_row79_col1" class="data row79 col1" >(1024,)</td>
      <td id="T_91102_row79_col2" class="data row79 col2" >1024</td>
      <td id="T_91102_row79_col3" class="data row79 col3" >1024</td>
      <td id="T_91102_row79_col4" class="data row79 col4" >(1024,)</td>
      <td id="T_91102_row79_col5" class="data row79 col5" >layer3.0.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row80" class="row_heading level0 row80" >80</th>
      <td id="T_91102_row80_col0" class="data row80 col0" >block_groups.2.block_group.0.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_91102_row80_col1" class="data row80 col1" >(1024,)</td>
      <td id="T_91102_row80_col2" class="data row80 col2" >1024</td>
      <td id="T_91102_row80_col3" class="data row80 col3" >1024</td>
      <td id="T_91102_row80_col4" class="data row80 col4" >(1024,)</td>
      <td id="T_91102_row80_col5" class="data row80 col5" >layer3.0.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row81" class="row_heading level0 row81" >81</th>
      <td id="T_91102_row81_col0" class="data row81 col0" >block_groups.2.block_group.0.right.conv.weight</td>
      <td id="T_91102_row81_col1" class="data row81 col1" >(1024, 512, 1, 1)</td>
      <td id="T_91102_row81_col2" class="data row81 col2" >524288</td>
      <td id="T_91102_row81_col3" class="data row81 col3" >524288</td>
      <td id="T_91102_row81_col4" class="data row81 col4" >(1024, 512, 1, 1)</td>
      <td id="T_91102_row81_col5" class="data row81 col5" >layer3.0.downsample.0.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row82" class="row_heading level0 row82" >82</th>
      <td id="T_91102_row82_col0" class="data row82 col0" >block_groups.2.block_group.0.right.batchnorm2d.weight</td>
      <td id="T_91102_row82_col1" class="data row82 col1" >(1024,)</td>
      <td id="T_91102_row82_col2" class="data row82 col2" >1024</td>
      <td id="T_91102_row82_col3" class="data row82 col3" >1024</td>
      <td id="T_91102_row82_col4" class="data row82 col4" >(1024,)</td>
      <td id="T_91102_row82_col5" class="data row82 col5" >layer3.0.downsample.1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row83" class="row_heading level0 row83" >83</th>
      <td id="T_91102_row83_col0" class="data row83 col0" >block_groups.2.block_group.0.right.batchnorm2d.bias</td>
      <td id="T_91102_row83_col1" class="data row83 col1" >(1024,)</td>
      <td id="T_91102_row83_col2" class="data row83 col2" >1024</td>
      <td id="T_91102_row83_col3" class="data row83 col3" >1024</td>
      <td id="T_91102_row83_col4" class="data row83 col4" >(1024,)</td>
      <td id="T_91102_row83_col5" class="data row83 col5" >layer3.0.downsample.1.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row84" class="row_heading level0 row84" >84</th>
      <td id="T_91102_row84_col0" class="data row84 col0" >block_groups.2.block_group.1.left.conv_block.0.conv.weight</td>
      <td id="T_91102_row84_col1" class="data row84 col1" >(256, 1024, 1, 1)</td>
      <td id="T_91102_row84_col2" class="data row84 col2" >262144</td>
      <td id="T_91102_row84_col3" class="data row84 col3" >262144</td>
      <td id="T_91102_row84_col4" class="data row84 col4" >(256, 1024, 1, 1)</td>
      <td id="T_91102_row84_col5" class="data row84 col5" >layer3.1.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row85" class="row_heading level0 row85" >85</th>
      <td id="T_91102_row85_col0" class="data row85 col0" >block_groups.2.block_group.1.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_91102_row85_col1" class="data row85 col1" >(256,)</td>
      <td id="T_91102_row85_col2" class="data row85 col2" >256</td>
      <td id="T_91102_row85_col3" class="data row85 col3" >256</td>
      <td id="T_91102_row85_col4" class="data row85 col4" >(256,)</td>
      <td id="T_91102_row85_col5" class="data row85 col5" >layer3.1.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row86" class="row_heading level0 row86" >86</th>
      <td id="T_91102_row86_col0" class="data row86 col0" >block_groups.2.block_group.1.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_91102_row86_col1" class="data row86 col1" >(256,)</td>
      <td id="T_91102_row86_col2" class="data row86 col2" >256</td>
      <td id="T_91102_row86_col3" class="data row86 col3" >256</td>
      <td id="T_91102_row86_col4" class="data row86 col4" >(256,)</td>
      <td id="T_91102_row86_col5" class="data row86 col5" >layer3.1.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row87" class="row_heading level0 row87" >87</th>
      <td id="T_91102_row87_col0" class="data row87 col0" >block_groups.2.block_group.1.left.conv_block.1.conv.weight</td>
      <td id="T_91102_row87_col1" class="data row87 col1" >(256, 256, 3, 3)</td>
      <td id="T_91102_row87_col2" class="data row87 col2" >589824</td>
      <td id="T_91102_row87_col3" class="data row87 col3" >589824</td>
      <td id="T_91102_row87_col4" class="data row87 col4" >(256, 256, 3, 3)</td>
      <td id="T_91102_row87_col5" class="data row87 col5" >layer3.1.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row88" class="row_heading level0 row88" >88</th>
      <td id="T_91102_row88_col0" class="data row88 col0" >block_groups.2.block_group.1.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_91102_row88_col1" class="data row88 col1" >(256,)</td>
      <td id="T_91102_row88_col2" class="data row88 col2" >256</td>
      <td id="T_91102_row88_col3" class="data row88 col3" >256</td>
      <td id="T_91102_row88_col4" class="data row88 col4" >(256,)</td>
      <td id="T_91102_row88_col5" class="data row88 col5" >layer3.1.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row89" class="row_heading level0 row89" >89</th>
      <td id="T_91102_row89_col0" class="data row89 col0" >block_groups.2.block_group.1.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_91102_row89_col1" class="data row89 col1" >(256,)</td>
      <td id="T_91102_row89_col2" class="data row89 col2" >256</td>
      <td id="T_91102_row89_col3" class="data row89 col3" >256</td>
      <td id="T_91102_row89_col4" class="data row89 col4" >(256,)</td>
      <td id="T_91102_row89_col5" class="data row89 col5" >layer3.1.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row90" class="row_heading level0 row90" >90</th>
      <td id="T_91102_row90_col0" class="data row90 col0" >block_groups.2.block_group.1.left.conv_block.2.conv.weight</td>
      <td id="T_91102_row90_col1" class="data row90 col1" >(1024, 256, 1, 1)</td>
      <td id="T_91102_row90_col2" class="data row90 col2" >262144</td>
      <td id="T_91102_row90_col3" class="data row90 col3" >262144</td>
      <td id="T_91102_row90_col4" class="data row90 col4" >(1024, 256, 1, 1)</td>
      <td id="T_91102_row90_col5" class="data row90 col5" >layer3.1.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row91" class="row_heading level0 row91" >91</th>
      <td id="T_91102_row91_col0" class="data row91 col0" >block_groups.2.block_group.1.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_91102_row91_col1" class="data row91 col1" >(1024,)</td>
      <td id="T_91102_row91_col2" class="data row91 col2" >1024</td>
      <td id="T_91102_row91_col3" class="data row91 col3" >1024</td>
      <td id="T_91102_row91_col4" class="data row91 col4" >(1024,)</td>
      <td id="T_91102_row91_col5" class="data row91 col5" >layer3.1.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row92" class="row_heading level0 row92" >92</th>
      <td id="T_91102_row92_col0" class="data row92 col0" >block_groups.2.block_group.1.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_91102_row92_col1" class="data row92 col1" >(1024,)</td>
      <td id="T_91102_row92_col2" class="data row92 col2" >1024</td>
      <td id="T_91102_row92_col3" class="data row92 col3" >1024</td>
      <td id="T_91102_row92_col4" class="data row92 col4" >(1024,)</td>
      <td id="T_91102_row92_col5" class="data row92 col5" >layer3.1.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row93" class="row_heading level0 row93" >93</th>
      <td id="T_91102_row93_col0" class="data row93 col0" >block_groups.2.block_group.2.left.conv_block.0.conv.weight</td>
      <td id="T_91102_row93_col1" class="data row93 col1" >(256, 1024, 1, 1)</td>
      <td id="T_91102_row93_col2" class="data row93 col2" >262144</td>
      <td id="T_91102_row93_col3" class="data row93 col3" >262144</td>
      <td id="T_91102_row93_col4" class="data row93 col4" >(256, 1024, 1, 1)</td>
      <td id="T_91102_row93_col5" class="data row93 col5" >layer3.2.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row94" class="row_heading level0 row94" >94</th>
      <td id="T_91102_row94_col0" class="data row94 col0" >block_groups.2.block_group.2.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_91102_row94_col1" class="data row94 col1" >(256,)</td>
      <td id="T_91102_row94_col2" class="data row94 col2" >256</td>
      <td id="T_91102_row94_col3" class="data row94 col3" >256</td>
      <td id="T_91102_row94_col4" class="data row94 col4" >(256,)</td>
      <td id="T_91102_row94_col5" class="data row94 col5" >layer3.2.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row95" class="row_heading level0 row95" >95</th>
      <td id="T_91102_row95_col0" class="data row95 col0" >block_groups.2.block_group.2.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_91102_row95_col1" class="data row95 col1" >(256,)</td>
      <td id="T_91102_row95_col2" class="data row95 col2" >256</td>
      <td id="T_91102_row95_col3" class="data row95 col3" >256</td>
      <td id="T_91102_row95_col4" class="data row95 col4" >(256,)</td>
      <td id="T_91102_row95_col5" class="data row95 col5" >layer3.2.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row96" class="row_heading level0 row96" >96</th>
      <td id="T_91102_row96_col0" class="data row96 col0" >block_groups.2.block_group.2.left.conv_block.1.conv.weight</td>
      <td id="T_91102_row96_col1" class="data row96 col1" >(256, 256, 3, 3)</td>
      <td id="T_91102_row96_col2" class="data row96 col2" >589824</td>
      <td id="T_91102_row96_col3" class="data row96 col3" >589824</td>
      <td id="T_91102_row96_col4" class="data row96 col4" >(256, 256, 3, 3)</td>
      <td id="T_91102_row96_col5" class="data row96 col5" >layer3.2.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row97" class="row_heading level0 row97" >97</th>
      <td id="T_91102_row97_col0" class="data row97 col0" >block_groups.2.block_group.2.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_91102_row97_col1" class="data row97 col1" >(256,)</td>
      <td id="T_91102_row97_col2" class="data row97 col2" >256</td>
      <td id="T_91102_row97_col3" class="data row97 col3" >256</td>
      <td id="T_91102_row97_col4" class="data row97 col4" >(256,)</td>
      <td id="T_91102_row97_col5" class="data row97 col5" >layer3.2.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row98" class="row_heading level0 row98" >98</th>
      <td id="T_91102_row98_col0" class="data row98 col0" >block_groups.2.block_group.2.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_91102_row98_col1" class="data row98 col1" >(256,)</td>
      <td id="T_91102_row98_col2" class="data row98 col2" >256</td>
      <td id="T_91102_row98_col3" class="data row98 col3" >256</td>
      <td id="T_91102_row98_col4" class="data row98 col4" >(256,)</td>
      <td id="T_91102_row98_col5" class="data row98 col5" >layer3.2.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row99" class="row_heading level0 row99" >99</th>
      <td id="T_91102_row99_col0" class="data row99 col0" >block_groups.2.block_group.2.left.conv_block.2.conv.weight</td>
      <td id="T_91102_row99_col1" class="data row99 col1" >(1024, 256, 1, 1)</td>
      <td id="T_91102_row99_col2" class="data row99 col2" >262144</td>
      <td id="T_91102_row99_col3" class="data row99 col3" >262144</td>
      <td id="T_91102_row99_col4" class="data row99 col4" >(1024, 256, 1, 1)</td>
      <td id="T_91102_row99_col5" class="data row99 col5" >layer3.2.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row100" class="row_heading level0 row100" >100</th>
      <td id="T_91102_row100_col0" class="data row100 col0" >block_groups.2.block_group.2.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_91102_row100_col1" class="data row100 col1" >(1024,)</td>
      <td id="T_91102_row100_col2" class="data row100 col2" >1024</td>
      <td id="T_91102_row100_col3" class="data row100 col3" >1024</td>
      <td id="T_91102_row100_col4" class="data row100 col4" >(1024,)</td>
      <td id="T_91102_row100_col5" class="data row100 col5" >layer3.2.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row101" class="row_heading level0 row101" >101</th>
      <td id="T_91102_row101_col0" class="data row101 col0" >block_groups.2.block_group.2.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_91102_row101_col1" class="data row101 col1" >(1024,)</td>
      <td id="T_91102_row101_col2" class="data row101 col2" >1024</td>
      <td id="T_91102_row101_col3" class="data row101 col3" >1024</td>
      <td id="T_91102_row101_col4" class="data row101 col4" >(1024,)</td>
      <td id="T_91102_row101_col5" class="data row101 col5" >layer3.2.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row102" class="row_heading level0 row102" >102</th>
      <td id="T_91102_row102_col0" class="data row102 col0" >block_groups.2.block_group.3.left.conv_block.0.conv.weight</td>
      <td id="T_91102_row102_col1" class="data row102 col1" >(256, 1024, 1, 1)</td>
      <td id="T_91102_row102_col2" class="data row102 col2" >262144</td>
      <td id="T_91102_row102_col3" class="data row102 col3" >262144</td>
      <td id="T_91102_row102_col4" class="data row102 col4" >(256, 1024, 1, 1)</td>
      <td id="T_91102_row102_col5" class="data row102 col5" >layer3.3.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row103" class="row_heading level0 row103" >103</th>
      <td id="T_91102_row103_col0" class="data row103 col0" >block_groups.2.block_group.3.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_91102_row103_col1" class="data row103 col1" >(256,)</td>
      <td id="T_91102_row103_col2" class="data row103 col2" >256</td>
      <td id="T_91102_row103_col3" class="data row103 col3" >256</td>
      <td id="T_91102_row103_col4" class="data row103 col4" >(256,)</td>
      <td id="T_91102_row103_col5" class="data row103 col5" >layer3.3.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row104" class="row_heading level0 row104" >104</th>
      <td id="T_91102_row104_col0" class="data row104 col0" >block_groups.2.block_group.3.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_91102_row104_col1" class="data row104 col1" >(256,)</td>
      <td id="T_91102_row104_col2" class="data row104 col2" >256</td>
      <td id="T_91102_row104_col3" class="data row104 col3" >256</td>
      <td id="T_91102_row104_col4" class="data row104 col4" >(256,)</td>
      <td id="T_91102_row104_col5" class="data row104 col5" >layer3.3.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row105" class="row_heading level0 row105" >105</th>
      <td id="T_91102_row105_col0" class="data row105 col0" >block_groups.2.block_group.3.left.conv_block.1.conv.weight</td>
      <td id="T_91102_row105_col1" class="data row105 col1" >(256, 256, 3, 3)</td>
      <td id="T_91102_row105_col2" class="data row105 col2" >589824</td>
      <td id="T_91102_row105_col3" class="data row105 col3" >589824</td>
      <td id="T_91102_row105_col4" class="data row105 col4" >(256, 256, 3, 3)</td>
      <td id="T_91102_row105_col5" class="data row105 col5" >layer3.3.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row106" class="row_heading level0 row106" >106</th>
      <td id="T_91102_row106_col0" class="data row106 col0" >block_groups.2.block_group.3.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_91102_row106_col1" class="data row106 col1" >(256,)</td>
      <td id="T_91102_row106_col2" class="data row106 col2" >256</td>
      <td id="T_91102_row106_col3" class="data row106 col3" >256</td>
      <td id="T_91102_row106_col4" class="data row106 col4" >(256,)</td>
      <td id="T_91102_row106_col5" class="data row106 col5" >layer3.3.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row107" class="row_heading level0 row107" >107</th>
      <td id="T_91102_row107_col0" class="data row107 col0" >block_groups.2.block_group.3.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_91102_row107_col1" class="data row107 col1" >(256,)</td>
      <td id="T_91102_row107_col2" class="data row107 col2" >256</td>
      <td id="T_91102_row107_col3" class="data row107 col3" >256</td>
      <td id="T_91102_row107_col4" class="data row107 col4" >(256,)</td>
      <td id="T_91102_row107_col5" class="data row107 col5" >layer3.3.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row108" class="row_heading level0 row108" >108</th>
      <td id="T_91102_row108_col0" class="data row108 col0" >block_groups.2.block_group.3.left.conv_block.2.conv.weight</td>
      <td id="T_91102_row108_col1" class="data row108 col1" >(1024, 256, 1, 1)</td>
      <td id="T_91102_row108_col2" class="data row108 col2" >262144</td>
      <td id="T_91102_row108_col3" class="data row108 col3" >262144</td>
      <td id="T_91102_row108_col4" class="data row108 col4" >(1024, 256, 1, 1)</td>
      <td id="T_91102_row108_col5" class="data row108 col5" >layer3.3.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row109" class="row_heading level0 row109" >109</th>
      <td id="T_91102_row109_col0" class="data row109 col0" >block_groups.2.block_group.3.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_91102_row109_col1" class="data row109 col1" >(1024,)</td>
      <td id="T_91102_row109_col2" class="data row109 col2" >1024</td>
      <td id="T_91102_row109_col3" class="data row109 col3" >1024</td>
      <td id="T_91102_row109_col4" class="data row109 col4" >(1024,)</td>
      <td id="T_91102_row109_col5" class="data row109 col5" >layer3.3.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row110" class="row_heading level0 row110" >110</th>
      <td id="T_91102_row110_col0" class="data row110 col0" >block_groups.2.block_group.3.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_91102_row110_col1" class="data row110 col1" >(1024,)</td>
      <td id="T_91102_row110_col2" class="data row110 col2" >1024</td>
      <td id="T_91102_row110_col3" class="data row110 col3" >1024</td>
      <td id="T_91102_row110_col4" class="data row110 col4" >(1024,)</td>
      <td id="T_91102_row110_col5" class="data row110 col5" >layer3.3.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row111" class="row_heading level0 row111" >111</th>
      <td id="T_91102_row111_col0" class="data row111 col0" >block_groups.2.block_group.4.left.conv_block.0.conv.weight</td>
      <td id="T_91102_row111_col1" class="data row111 col1" >(256, 1024, 1, 1)</td>
      <td id="T_91102_row111_col2" class="data row111 col2" >262144</td>
      <td id="T_91102_row111_col3" class="data row111 col3" >262144</td>
      <td id="T_91102_row111_col4" class="data row111 col4" >(256, 1024, 1, 1)</td>
      <td id="T_91102_row111_col5" class="data row111 col5" >layer3.4.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row112" class="row_heading level0 row112" >112</th>
      <td id="T_91102_row112_col0" class="data row112 col0" >block_groups.2.block_group.4.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_91102_row112_col1" class="data row112 col1" >(256,)</td>
      <td id="T_91102_row112_col2" class="data row112 col2" >256</td>
      <td id="T_91102_row112_col3" class="data row112 col3" >256</td>
      <td id="T_91102_row112_col4" class="data row112 col4" >(256,)</td>
      <td id="T_91102_row112_col5" class="data row112 col5" >layer3.4.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row113" class="row_heading level0 row113" >113</th>
      <td id="T_91102_row113_col0" class="data row113 col0" >block_groups.2.block_group.4.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_91102_row113_col1" class="data row113 col1" >(256,)</td>
      <td id="T_91102_row113_col2" class="data row113 col2" >256</td>
      <td id="T_91102_row113_col3" class="data row113 col3" >256</td>
      <td id="T_91102_row113_col4" class="data row113 col4" >(256,)</td>
      <td id="T_91102_row113_col5" class="data row113 col5" >layer3.4.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row114" class="row_heading level0 row114" >114</th>
      <td id="T_91102_row114_col0" class="data row114 col0" >block_groups.2.block_group.4.left.conv_block.1.conv.weight</td>
      <td id="T_91102_row114_col1" class="data row114 col1" >(256, 256, 3, 3)</td>
      <td id="T_91102_row114_col2" class="data row114 col2" >589824</td>
      <td id="T_91102_row114_col3" class="data row114 col3" >589824</td>
      <td id="T_91102_row114_col4" class="data row114 col4" >(256, 256, 3, 3)</td>
      <td id="T_91102_row114_col5" class="data row114 col5" >layer3.4.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row115" class="row_heading level0 row115" >115</th>
      <td id="T_91102_row115_col0" class="data row115 col0" >block_groups.2.block_group.4.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_91102_row115_col1" class="data row115 col1" >(256,)</td>
      <td id="T_91102_row115_col2" class="data row115 col2" >256</td>
      <td id="T_91102_row115_col3" class="data row115 col3" >256</td>
      <td id="T_91102_row115_col4" class="data row115 col4" >(256,)</td>
      <td id="T_91102_row115_col5" class="data row115 col5" >layer3.4.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row116" class="row_heading level0 row116" >116</th>
      <td id="T_91102_row116_col0" class="data row116 col0" >block_groups.2.block_group.4.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_91102_row116_col1" class="data row116 col1" >(256,)</td>
      <td id="T_91102_row116_col2" class="data row116 col2" >256</td>
      <td id="T_91102_row116_col3" class="data row116 col3" >256</td>
      <td id="T_91102_row116_col4" class="data row116 col4" >(256,)</td>
      <td id="T_91102_row116_col5" class="data row116 col5" >layer3.4.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row117" class="row_heading level0 row117" >117</th>
      <td id="T_91102_row117_col0" class="data row117 col0" >block_groups.2.block_group.4.left.conv_block.2.conv.weight</td>
      <td id="T_91102_row117_col1" class="data row117 col1" >(1024, 256, 1, 1)</td>
      <td id="T_91102_row117_col2" class="data row117 col2" >262144</td>
      <td id="T_91102_row117_col3" class="data row117 col3" >262144</td>
      <td id="T_91102_row117_col4" class="data row117 col4" >(1024, 256, 1, 1)</td>
      <td id="T_91102_row117_col5" class="data row117 col5" >layer3.4.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row118" class="row_heading level0 row118" >118</th>
      <td id="T_91102_row118_col0" class="data row118 col0" >block_groups.2.block_group.4.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_91102_row118_col1" class="data row118 col1" >(1024,)</td>
      <td id="T_91102_row118_col2" class="data row118 col2" >1024</td>
      <td id="T_91102_row118_col3" class="data row118 col3" >1024</td>
      <td id="T_91102_row118_col4" class="data row118 col4" >(1024,)</td>
      <td id="T_91102_row118_col5" class="data row118 col5" >layer3.4.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row119" class="row_heading level0 row119" >119</th>
      <td id="T_91102_row119_col0" class="data row119 col0" >block_groups.2.block_group.4.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_91102_row119_col1" class="data row119 col1" >(1024,)</td>
      <td id="T_91102_row119_col2" class="data row119 col2" >1024</td>
      <td id="T_91102_row119_col3" class="data row119 col3" >1024</td>
      <td id="T_91102_row119_col4" class="data row119 col4" >(1024,)</td>
      <td id="T_91102_row119_col5" class="data row119 col5" >layer3.4.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row120" class="row_heading level0 row120" >120</th>
      <td id="T_91102_row120_col0" class="data row120 col0" >block_groups.2.block_group.5.left.conv_block.0.conv.weight</td>
      <td id="T_91102_row120_col1" class="data row120 col1" >(256, 1024, 1, 1)</td>
      <td id="T_91102_row120_col2" class="data row120 col2" >262144</td>
      <td id="T_91102_row120_col3" class="data row120 col3" >262144</td>
      <td id="T_91102_row120_col4" class="data row120 col4" >(256, 1024, 1, 1)</td>
      <td id="T_91102_row120_col5" class="data row120 col5" >layer3.5.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row121" class="row_heading level0 row121" >121</th>
      <td id="T_91102_row121_col0" class="data row121 col0" >block_groups.2.block_group.5.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_91102_row121_col1" class="data row121 col1" >(256,)</td>
      <td id="T_91102_row121_col2" class="data row121 col2" >256</td>
      <td id="T_91102_row121_col3" class="data row121 col3" >256</td>
      <td id="T_91102_row121_col4" class="data row121 col4" >(256,)</td>
      <td id="T_91102_row121_col5" class="data row121 col5" >layer3.5.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row122" class="row_heading level0 row122" >122</th>
      <td id="T_91102_row122_col0" class="data row122 col0" >block_groups.2.block_group.5.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_91102_row122_col1" class="data row122 col1" >(256,)</td>
      <td id="T_91102_row122_col2" class="data row122 col2" >256</td>
      <td id="T_91102_row122_col3" class="data row122 col3" >256</td>
      <td id="T_91102_row122_col4" class="data row122 col4" >(256,)</td>
      <td id="T_91102_row122_col5" class="data row122 col5" >layer3.5.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row123" class="row_heading level0 row123" >123</th>
      <td id="T_91102_row123_col0" class="data row123 col0" >block_groups.2.block_group.5.left.conv_block.1.conv.weight</td>
      <td id="T_91102_row123_col1" class="data row123 col1" >(256, 256, 3, 3)</td>
      <td id="T_91102_row123_col2" class="data row123 col2" >589824</td>
      <td id="T_91102_row123_col3" class="data row123 col3" >589824</td>
      <td id="T_91102_row123_col4" class="data row123 col4" >(256, 256, 3, 3)</td>
      <td id="T_91102_row123_col5" class="data row123 col5" >layer3.5.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row124" class="row_heading level0 row124" >124</th>
      <td id="T_91102_row124_col0" class="data row124 col0" >block_groups.2.block_group.5.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_91102_row124_col1" class="data row124 col1" >(256,)</td>
      <td id="T_91102_row124_col2" class="data row124 col2" >256</td>
      <td id="T_91102_row124_col3" class="data row124 col3" >256</td>
      <td id="T_91102_row124_col4" class="data row124 col4" >(256,)</td>
      <td id="T_91102_row124_col5" class="data row124 col5" >layer3.5.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row125" class="row_heading level0 row125" >125</th>
      <td id="T_91102_row125_col0" class="data row125 col0" >block_groups.2.block_group.5.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_91102_row125_col1" class="data row125 col1" >(256,)</td>
      <td id="T_91102_row125_col2" class="data row125 col2" >256</td>
      <td id="T_91102_row125_col3" class="data row125 col3" >256</td>
      <td id="T_91102_row125_col4" class="data row125 col4" >(256,)</td>
      <td id="T_91102_row125_col5" class="data row125 col5" >layer3.5.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row126" class="row_heading level0 row126" >126</th>
      <td id="T_91102_row126_col0" class="data row126 col0" >block_groups.2.block_group.5.left.conv_block.2.conv.weight</td>
      <td id="T_91102_row126_col1" class="data row126 col1" >(1024, 256, 1, 1)</td>
      <td id="T_91102_row126_col2" class="data row126 col2" >262144</td>
      <td id="T_91102_row126_col3" class="data row126 col3" >262144</td>
      <td id="T_91102_row126_col4" class="data row126 col4" >(1024, 256, 1, 1)</td>
      <td id="T_91102_row126_col5" class="data row126 col5" >layer3.5.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row127" class="row_heading level0 row127" >127</th>
      <td id="T_91102_row127_col0" class="data row127 col0" >block_groups.2.block_group.5.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_91102_row127_col1" class="data row127 col1" >(1024,)</td>
      <td id="T_91102_row127_col2" class="data row127 col2" >1024</td>
      <td id="T_91102_row127_col3" class="data row127 col3" >1024</td>
      <td id="T_91102_row127_col4" class="data row127 col4" >(1024,)</td>
      <td id="T_91102_row127_col5" class="data row127 col5" >layer3.5.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row128" class="row_heading level0 row128" >128</th>
      <td id="T_91102_row128_col0" class="data row128 col0" >block_groups.2.block_group.5.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_91102_row128_col1" class="data row128 col1" >(1024,)</td>
      <td id="T_91102_row128_col2" class="data row128 col2" >1024</td>
      <td id="T_91102_row128_col3" class="data row128 col3" >1024</td>
      <td id="T_91102_row128_col4" class="data row128 col4" >(1024,)</td>
      <td id="T_91102_row128_col5" class="data row128 col5" >layer3.5.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row129" class="row_heading level0 row129" >129</th>
      <td id="T_91102_row129_col0" class="data row129 col0" >block_groups.3.block_group.0.left.conv_block.0.conv.weight</td>
      <td id="T_91102_row129_col1" class="data row129 col1" >(512, 1024, 1, 1)</td>
      <td id="T_91102_row129_col2" class="data row129 col2" >524288</td>
      <td id="T_91102_row129_col3" class="data row129 col3" >524288</td>
      <td id="T_91102_row129_col4" class="data row129 col4" >(512, 1024, 1, 1)</td>
      <td id="T_91102_row129_col5" class="data row129 col5" >layer4.0.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row130" class="row_heading level0 row130" >130</th>
      <td id="T_91102_row130_col0" class="data row130 col0" >block_groups.3.block_group.0.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_91102_row130_col1" class="data row130 col1" >(512,)</td>
      <td id="T_91102_row130_col2" class="data row130 col2" >512</td>
      <td id="T_91102_row130_col3" class="data row130 col3" >512</td>
      <td id="T_91102_row130_col4" class="data row130 col4" >(512,)</td>
      <td id="T_91102_row130_col5" class="data row130 col5" >layer4.0.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row131" class="row_heading level0 row131" >131</th>
      <td id="T_91102_row131_col0" class="data row131 col0" >block_groups.3.block_group.0.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_91102_row131_col1" class="data row131 col1" >(512,)</td>
      <td id="T_91102_row131_col2" class="data row131 col2" >512</td>
      <td id="T_91102_row131_col3" class="data row131 col3" >512</td>
      <td id="T_91102_row131_col4" class="data row131 col4" >(512,)</td>
      <td id="T_91102_row131_col5" class="data row131 col5" >layer4.0.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row132" class="row_heading level0 row132" >132</th>
      <td id="T_91102_row132_col0" class="data row132 col0" >block_groups.3.block_group.0.left.conv_block.1.conv.weight</td>
      <td id="T_91102_row132_col1" class="data row132 col1" >(512, 512, 3, 3)</td>
      <td id="T_91102_row132_col2" class="data row132 col2" >2359296</td>
      <td id="T_91102_row132_col3" class="data row132 col3" >2359296</td>
      <td id="T_91102_row132_col4" class="data row132 col4" >(512, 512, 3, 3)</td>
      <td id="T_91102_row132_col5" class="data row132 col5" >layer4.0.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row133" class="row_heading level0 row133" >133</th>
      <td id="T_91102_row133_col0" class="data row133 col0" >block_groups.3.block_group.0.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_91102_row133_col1" class="data row133 col1" >(512,)</td>
      <td id="T_91102_row133_col2" class="data row133 col2" >512</td>
      <td id="T_91102_row133_col3" class="data row133 col3" >512</td>
      <td id="T_91102_row133_col4" class="data row133 col4" >(512,)</td>
      <td id="T_91102_row133_col5" class="data row133 col5" >layer4.0.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row134" class="row_heading level0 row134" >134</th>
      <td id="T_91102_row134_col0" class="data row134 col0" >block_groups.3.block_group.0.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_91102_row134_col1" class="data row134 col1" >(512,)</td>
      <td id="T_91102_row134_col2" class="data row134 col2" >512</td>
      <td id="T_91102_row134_col3" class="data row134 col3" >512</td>
      <td id="T_91102_row134_col4" class="data row134 col4" >(512,)</td>
      <td id="T_91102_row134_col5" class="data row134 col5" >layer4.0.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row135" class="row_heading level0 row135" >135</th>
      <td id="T_91102_row135_col0" class="data row135 col0" >block_groups.3.block_group.0.left.conv_block.2.conv.weight</td>
      <td id="T_91102_row135_col1" class="data row135 col1" >(2048, 512, 1, 1)</td>
      <td id="T_91102_row135_col2" class="data row135 col2" >1048576</td>
      <td id="T_91102_row135_col3" class="data row135 col3" >1048576</td>
      <td id="T_91102_row135_col4" class="data row135 col4" >(2048, 512, 1, 1)</td>
      <td id="T_91102_row135_col5" class="data row135 col5" >layer4.0.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row136" class="row_heading level0 row136" >136</th>
      <td id="T_91102_row136_col0" class="data row136 col0" >block_groups.3.block_group.0.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_91102_row136_col1" class="data row136 col1" >(2048,)</td>
      <td id="T_91102_row136_col2" class="data row136 col2" >2048</td>
      <td id="T_91102_row136_col3" class="data row136 col3" >2048</td>
      <td id="T_91102_row136_col4" class="data row136 col4" >(2048,)</td>
      <td id="T_91102_row136_col5" class="data row136 col5" >layer4.0.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row137" class="row_heading level0 row137" >137</th>
      <td id="T_91102_row137_col0" class="data row137 col0" >block_groups.3.block_group.0.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_91102_row137_col1" class="data row137 col1" >(2048,)</td>
      <td id="T_91102_row137_col2" class="data row137 col2" >2048</td>
      <td id="T_91102_row137_col3" class="data row137 col3" >2048</td>
      <td id="T_91102_row137_col4" class="data row137 col4" >(2048,)</td>
      <td id="T_91102_row137_col5" class="data row137 col5" >layer4.0.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row138" class="row_heading level0 row138" >138</th>
      <td id="T_91102_row138_col0" class="data row138 col0" >block_groups.3.block_group.0.right.conv.weight</td>
      <td id="T_91102_row138_col1" class="data row138 col1" >(2048, 1024, 1, 1)</td>
      <td id="T_91102_row138_col2" class="data row138 col2" >2097152</td>
      <td id="T_91102_row138_col3" class="data row138 col3" >2097152</td>
      <td id="T_91102_row138_col4" class="data row138 col4" >(2048, 1024, 1, 1)</td>
      <td id="T_91102_row138_col5" class="data row138 col5" >layer4.0.downsample.0.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row139" class="row_heading level0 row139" >139</th>
      <td id="T_91102_row139_col0" class="data row139 col0" >block_groups.3.block_group.0.right.batchnorm2d.weight</td>
      <td id="T_91102_row139_col1" class="data row139 col1" >(2048,)</td>
      <td id="T_91102_row139_col2" class="data row139 col2" >2048</td>
      <td id="T_91102_row139_col3" class="data row139 col3" >2048</td>
      <td id="T_91102_row139_col4" class="data row139 col4" >(2048,)</td>
      <td id="T_91102_row139_col5" class="data row139 col5" >layer4.0.downsample.1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row140" class="row_heading level0 row140" >140</th>
      <td id="T_91102_row140_col0" class="data row140 col0" >block_groups.3.block_group.0.right.batchnorm2d.bias</td>
      <td id="T_91102_row140_col1" class="data row140 col1" >(2048,)</td>
      <td id="T_91102_row140_col2" class="data row140 col2" >2048</td>
      <td id="T_91102_row140_col3" class="data row140 col3" >2048</td>
      <td id="T_91102_row140_col4" class="data row140 col4" >(2048,)</td>
      <td id="T_91102_row140_col5" class="data row140 col5" >layer4.0.downsample.1.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row141" class="row_heading level0 row141" >141</th>
      <td id="T_91102_row141_col0" class="data row141 col0" >block_groups.3.block_group.1.left.conv_block.0.conv.weight</td>
      <td id="T_91102_row141_col1" class="data row141 col1" >(512, 2048, 1, 1)</td>
      <td id="T_91102_row141_col2" class="data row141 col2" >1048576</td>
      <td id="T_91102_row141_col3" class="data row141 col3" >1048576</td>
      <td id="T_91102_row141_col4" class="data row141 col4" >(512, 2048, 1, 1)</td>
      <td id="T_91102_row141_col5" class="data row141 col5" >layer4.1.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row142" class="row_heading level0 row142" >142</th>
      <td id="T_91102_row142_col0" class="data row142 col0" >block_groups.3.block_group.1.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_91102_row142_col1" class="data row142 col1" >(512,)</td>
      <td id="T_91102_row142_col2" class="data row142 col2" >512</td>
      <td id="T_91102_row142_col3" class="data row142 col3" >512</td>
      <td id="T_91102_row142_col4" class="data row142 col4" >(512,)</td>
      <td id="T_91102_row142_col5" class="data row142 col5" >layer4.1.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row143" class="row_heading level0 row143" >143</th>
      <td id="T_91102_row143_col0" class="data row143 col0" >block_groups.3.block_group.1.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_91102_row143_col1" class="data row143 col1" >(512,)</td>
      <td id="T_91102_row143_col2" class="data row143 col2" >512</td>
      <td id="T_91102_row143_col3" class="data row143 col3" >512</td>
      <td id="T_91102_row143_col4" class="data row143 col4" >(512,)</td>
      <td id="T_91102_row143_col5" class="data row143 col5" >layer4.1.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row144" class="row_heading level0 row144" >144</th>
      <td id="T_91102_row144_col0" class="data row144 col0" >block_groups.3.block_group.1.left.conv_block.1.conv.weight</td>
      <td id="T_91102_row144_col1" class="data row144 col1" >(512, 512, 3, 3)</td>
      <td id="T_91102_row144_col2" class="data row144 col2" >2359296</td>
      <td id="T_91102_row144_col3" class="data row144 col3" >2359296</td>
      <td id="T_91102_row144_col4" class="data row144 col4" >(512, 512, 3, 3)</td>
      <td id="T_91102_row144_col5" class="data row144 col5" >layer4.1.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row145" class="row_heading level0 row145" >145</th>
      <td id="T_91102_row145_col0" class="data row145 col0" >block_groups.3.block_group.1.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_91102_row145_col1" class="data row145 col1" >(512,)</td>
      <td id="T_91102_row145_col2" class="data row145 col2" >512</td>
      <td id="T_91102_row145_col3" class="data row145 col3" >512</td>
      <td id="T_91102_row145_col4" class="data row145 col4" >(512,)</td>
      <td id="T_91102_row145_col5" class="data row145 col5" >layer4.1.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row146" class="row_heading level0 row146" >146</th>
      <td id="T_91102_row146_col0" class="data row146 col0" >block_groups.3.block_group.1.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_91102_row146_col1" class="data row146 col1" >(512,)</td>
      <td id="T_91102_row146_col2" class="data row146 col2" >512</td>
      <td id="T_91102_row146_col3" class="data row146 col3" >512</td>
      <td id="T_91102_row146_col4" class="data row146 col4" >(512,)</td>
      <td id="T_91102_row146_col5" class="data row146 col5" >layer4.1.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row147" class="row_heading level0 row147" >147</th>
      <td id="T_91102_row147_col0" class="data row147 col0" >block_groups.3.block_group.1.left.conv_block.2.conv.weight</td>
      <td id="T_91102_row147_col1" class="data row147 col1" >(2048, 512, 1, 1)</td>
      <td id="T_91102_row147_col2" class="data row147 col2" >1048576</td>
      <td id="T_91102_row147_col3" class="data row147 col3" >1048576</td>
      <td id="T_91102_row147_col4" class="data row147 col4" >(2048, 512, 1, 1)</td>
      <td id="T_91102_row147_col5" class="data row147 col5" >layer4.1.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row148" class="row_heading level0 row148" >148</th>
      <td id="T_91102_row148_col0" class="data row148 col0" >block_groups.3.block_group.1.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_91102_row148_col1" class="data row148 col1" >(2048,)</td>
      <td id="T_91102_row148_col2" class="data row148 col2" >2048</td>
      <td id="T_91102_row148_col3" class="data row148 col3" >2048</td>
      <td id="T_91102_row148_col4" class="data row148 col4" >(2048,)</td>
      <td id="T_91102_row148_col5" class="data row148 col5" >layer4.1.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row149" class="row_heading level0 row149" >149</th>
      <td id="T_91102_row149_col0" class="data row149 col0" >block_groups.3.block_group.1.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_91102_row149_col1" class="data row149 col1" >(2048,)</td>
      <td id="T_91102_row149_col2" class="data row149 col2" >2048</td>
      <td id="T_91102_row149_col3" class="data row149 col3" >2048</td>
      <td id="T_91102_row149_col4" class="data row149 col4" >(2048,)</td>
      <td id="T_91102_row149_col5" class="data row149 col5" >layer4.1.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row150" class="row_heading level0 row150" >150</th>
      <td id="T_91102_row150_col0" class="data row150 col0" >block_groups.3.block_group.2.left.conv_block.0.conv.weight</td>
      <td id="T_91102_row150_col1" class="data row150 col1" >(512, 2048, 1, 1)</td>
      <td id="T_91102_row150_col2" class="data row150 col2" >1048576</td>
      <td id="T_91102_row150_col3" class="data row150 col3" >1048576</td>
      <td id="T_91102_row150_col4" class="data row150 col4" >(512, 2048, 1, 1)</td>
      <td id="T_91102_row150_col5" class="data row150 col5" >layer4.2.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row151" class="row_heading level0 row151" >151</th>
      <td id="T_91102_row151_col0" class="data row151 col0" >block_groups.3.block_group.2.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_91102_row151_col1" class="data row151 col1" >(512,)</td>
      <td id="T_91102_row151_col2" class="data row151 col2" >512</td>
      <td id="T_91102_row151_col3" class="data row151 col3" >512</td>
      <td id="T_91102_row151_col4" class="data row151 col4" >(512,)</td>
      <td id="T_91102_row151_col5" class="data row151 col5" >layer4.2.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row152" class="row_heading level0 row152" >152</th>
      <td id="T_91102_row152_col0" class="data row152 col0" >block_groups.3.block_group.2.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_91102_row152_col1" class="data row152 col1" >(512,)</td>
      <td id="T_91102_row152_col2" class="data row152 col2" >512</td>
      <td id="T_91102_row152_col3" class="data row152 col3" >512</td>
      <td id="T_91102_row152_col4" class="data row152 col4" >(512,)</td>
      <td id="T_91102_row152_col5" class="data row152 col5" >layer4.2.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row153" class="row_heading level0 row153" >153</th>
      <td id="T_91102_row153_col0" class="data row153 col0" >block_groups.3.block_group.2.left.conv_block.1.conv.weight</td>
      <td id="T_91102_row153_col1" class="data row153 col1" >(512, 512, 3, 3)</td>
      <td id="T_91102_row153_col2" class="data row153 col2" >2359296</td>
      <td id="T_91102_row153_col3" class="data row153 col3" >2359296</td>
      <td id="T_91102_row153_col4" class="data row153 col4" >(512, 512, 3, 3)</td>
      <td id="T_91102_row153_col5" class="data row153 col5" >layer4.2.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row154" class="row_heading level0 row154" >154</th>
      <td id="T_91102_row154_col0" class="data row154 col0" >block_groups.3.block_group.2.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_91102_row154_col1" class="data row154 col1" >(512,)</td>
      <td id="T_91102_row154_col2" class="data row154 col2" >512</td>
      <td id="T_91102_row154_col3" class="data row154 col3" >512</td>
      <td id="T_91102_row154_col4" class="data row154 col4" >(512,)</td>
      <td id="T_91102_row154_col5" class="data row154 col5" >layer4.2.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row155" class="row_heading level0 row155" >155</th>
      <td id="T_91102_row155_col0" class="data row155 col0" >block_groups.3.block_group.2.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_91102_row155_col1" class="data row155 col1" >(512,)</td>
      <td id="T_91102_row155_col2" class="data row155 col2" >512</td>
      <td id="T_91102_row155_col3" class="data row155 col3" >512</td>
      <td id="T_91102_row155_col4" class="data row155 col4" >(512,)</td>
      <td id="T_91102_row155_col5" class="data row155 col5" >layer4.2.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row156" class="row_heading level0 row156" >156</th>
      <td id="T_91102_row156_col0" class="data row156 col0" >block_groups.3.block_group.2.left.conv_block.2.conv.weight</td>
      <td id="T_91102_row156_col1" class="data row156 col1" >(2048, 512, 1, 1)</td>
      <td id="T_91102_row156_col2" class="data row156 col2" >1048576</td>
      <td id="T_91102_row156_col3" class="data row156 col3" >1048576</td>
      <td id="T_91102_row156_col4" class="data row156 col4" >(2048, 512, 1, 1)</td>
      <td id="T_91102_row156_col5" class="data row156 col5" >layer4.2.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row157" class="row_heading level0 row157" >157</th>
      <td id="T_91102_row157_col0" class="data row157 col0" >block_groups.3.block_group.2.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_91102_row157_col1" class="data row157 col1" >(2048,)</td>
      <td id="T_91102_row157_col2" class="data row157 col2" >2048</td>
      <td id="T_91102_row157_col3" class="data row157 col3" >2048</td>
      <td id="T_91102_row157_col4" class="data row157 col4" >(2048,)</td>
      <td id="T_91102_row157_col5" class="data row157 col5" >layer4.2.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row158" class="row_heading level0 row158" >158</th>
      <td id="T_91102_row158_col0" class="data row158 col0" >block_groups.3.block_group.2.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_91102_row158_col1" class="data row158 col1" >(2048,)</td>
      <td id="T_91102_row158_col2" class="data row158 col2" >2048</td>
      <td id="T_91102_row158_col3" class="data row158 col3" >2048</td>
      <td id="T_91102_row158_col4" class="data row158 col4" >(2048,)</td>
      <td id="T_91102_row158_col5" class="data row158 col5" >layer4.2.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row159" class="row_heading level0 row159" >159</th>
      <td id="T_91102_row159_col0" class="data row159 col0" >output_layers.2.weight</td>
      <td id="T_91102_row159_col1" class="data row159 col1" >(1000, 2048)</td>
      <td id="T_91102_row159_col2" class="data row159 col2" >2048000</td>
      <td id="T_91102_row159_col3" class="data row159 col3" >2048000</td>
      <td id="T_91102_row159_col4" class="data row159 col4" >(1000, 2048)</td>
      <td id="T_91102_row159_col5" class="data row159 col5" >fc.weight</td>
    </tr>
    <tr>
      <th id="T_91102_level0_row160" class="row_heading level0 row160" >160</th>
      <td id="T_91102_row160_col0" class="data row160 col0" >output_layers.2.bias</td>
      <td id="T_91102_row160_col1" class="data row160 col1" >(1000,)</td>
      <td id="T_91102_row160_col2" class="data row160 col2" >1000</td>
      <td id="T_91102_row160_col3" class="data row160 col3" >1000</td>
      <td id="T_91102_row160_col4" class="data row160 col4" >(1000,)</td>
      <td id="T_91102_row160_col5" class="data row160 col5" >fc.bias</td>
    </tr>
  </tbody>
</table>



####  Classify some RGB images

##### Helper functions

We follow https://pytorch.org/hub/pytorch_vision_resnet/ for preprocessing images, using a pretrained model (as we did in the previous section), and looking up human-readable classes from model predictions.


```python
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```


```python
def predict_top5(model, image_filenames):
  model.eval()
  images = [PIL.Image.open(filename) for filename in image_filenames]
  input_batch = t.stack([preprocess(img) for img in images], dim=0)

  # move the input and model to GPU for speed if available
  if t.cuda.is_available():
      input_batch = input_batch.to('cuda')
      model.to('cuda')

  # run the model
  with t.no_grad():
      output = model(input_batch)

  probabilities = t.nn.functional.softmax(output, dim=1)

  # Read the categories
  with open("imagenet_classes.txt", "r") as f:
      categories = [s.strip() for s in f.readlines()]

  # Show top categories per image
  top5_prob, top5_catid = t.topk(probabilities, k=5)
  for i in range(top5_prob.size(0)):
    display(images[i])
    for j in range(top5_prob.size(1)):
      print(categories[top5_catid[i][j]], top5_prob[i][j].item())
    print()
```

Even though two models can have the same parameters, that doesn't mean the architectures are identical. Things like where a stride is taken in a `BlockGroup` can have an impact on model output. We can check the equality of two models by checking the equality of their outputs.


```python
def compare_predictions(my_model, pretrained_model, image_filenames, atol=0., rtol=0.):
  my_model.eval()
  pretrained_model.eval()

  images = [PIL.Image.open(filename) for filename in image_filenames]
  input_batch = t.stack([preprocess(img) for img in images], dim=0)

  # move the input and model to GPU for speed if available
  if t.cuda.is_available():
    input_batch = input_batch.to('cuda')
    my_model.to('cuda')
    pretrained_model.to('cuda')

  # run the models
  with t.no_grad():
    output_my_model = my_model(input_batch)
    output_pretrained_model = pretrained_model(input_batch)
    prob_my_model = t.nn.functional.softmax(output_my_model, dim=1)
    prob_pretrained_model = t.nn.functional.softmax(output_pretrained_model, dim=1)

  if t.allclose(prob_my_model, prob_pretrained_model, atol, rtol):
    print("Models are equivalent!")
  else:
    print("Models produce different outputs. Check architecture implementation.")


```

##### Classify some images and compare outputs

We imported some images during the "Install dependencies" step and can use them here.


```python
folder_path = "test_images/"
IMAGE_NAMES = ['golden_retriever_puppy.jpg', 'grizzly_bear.jpg', 'golden_gate_bridge.jpg', 'general_sherman_tree.jpg', 'muni_train.jpg']
IMAGE_FILENAMES = [folder_path + image_name for image_name in IMAGE_NAMES]
```

Now we can classify some images to check our architecture implementation.

###### Test ResNet34


```python
compare_predictions(my_resnet34, pretrained_resnet34, IMAGE_FILENAMES, 1e-5)
```

    Models are equivalent!


###### Test ResNet50


```python
compare_predictions(my_resnet50, pretrained_resnet50, IMAGE_FILENAMES, 1e-5)
```

    Models are equivalent!


Let's also see what the top 5 predicted classes are for each image, just for fun.


```python
predict_top5(my_resnet50, IMAGE_FILENAMES)
```


    
<img src="/assets/img/resnet/resnet_from_scratch_67_0.png">
    


    golden retriever 0.9919309020042419
    Brittany spaniel 0.0028146819677203894
    Labrador retriever 0.0012666297843679786
    tennis ball 0.0006666457047685981
    kuvasz 0.0005991900688968599
    



    
<img src="/assets/img/resnet/resnet_from_scratch_67_2.png">
    


    brown bear 0.9999508857727051
    American black bear 4.4745906052412465e-05
    ice bear 3.734054644155549e-06
    sloth bear 5.089370347377553e-07
    bison 4.709042400463659e-08
    



    
<img src="/assets/img/resnet/resnet_from_scratch_67_4.png">
    


    suspension bridge 0.6104913353919983
    pier 0.3443833291530609
    steel arch bridge 0.02683708816766739
    promontory 0.009572144597768784
    viaduct 0.0035211530048400164
    



    
<img src="/assets/img/resnet/resnet_from_scratch_67_6.png">
    


    obelisk 0.6610313653945923
    megalith 0.10190818458795547
    pole 0.06568432599306107
    totem pole 0.024245884269475937
    fountain 0.0200052410364151
    



    
<img src="/assets/img/resnet/resnet_from_scratch_67_8.png">
    


    streetcar 0.9358044266700745
    trolleybus 0.06182621046900749
    passenger car 0.0013690213672816753
    electric locomotive 0.000848782598040998
    bullet train 3.628083140938543e-05
    


# Replacing PyTorch building blocks with our own by subclassing nn.Module

`torch.nn` Modules are stateful building blocks that implement the forward pass of a computation (by implementing `forward()`) and partner with PyTorch's autograd system for the backward pass. Parameters to be learned in a module are specified by wrapping their tensors in a `torch.nn.Parameter`, which automatically registers them with the module to be updated by forward/backward passes.

All modules should subclass `torch.nn.Module` to be composable with other modules. The submodules of a module can be accessed via calls to `children()` or `named_children()`. To be able to recursively access all the submodules' children, call `modules()` or `named_modules()`.

In order to dynamically define submodules `ModuleList` or `ModuleDict` can be used to register submodules from a list or a dict. Another way is what we did above, with `Sequential(*args)`.

`parameters()` or `named_parameters()` can be used to recursively access all parameters of a module and its submodules.

A general function to recursively apply any function to a module and its submodules is `apply()`.

Modules have a training mode and evaluation mode, which can be toggled between with `train()` and `eval()`. Some modules, like `BatchNorm2d`, have different behavior depending on its modality.

A trained model's state can be saved to disk by calling its `state_dict()` and loaded with `load_state_dict()`.

State includes not only modules' parameters, but also any "persistent buffers" for non-learnable aspects of computation, like the running mean and variance in a `BatchNorm2d` layer. Non-persistent buffers are not saved. Buffers can be registered via `register_buffer()`, which marks them as persistent by default. Buffers can be accessed recursively, unsurprisingly at this point, with `buffers()` or `named_buffers()`.

For more details and other features of `torch.nn.Module`, consult https://pytorch.org/docs/stable/notes/modules.html.


### Implementing our own modules with torch.nn.Module


```python
class CustomReLU(t.nn.Module):
  def forward(self, x):
    return t.maximum(x, t.tensor(0.))
```


```python
class CustomLinear(t.nn.Module):
  def __init__(self, in_features, out_features, bias=True):
    '''
    A simple linear (technically, affine) transformation.

    The fields should be named `weight` and `bias` for compatibility with
    PyTorch.
    If `bias` is False, set `self.bias` to None.
    '''
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    weight = 2*(t.rand(out_features, in_features) - 0.5) / \
              t.sqrt(t.tensor(in_features))
    self.weight = t.nn.Parameter(weight)
    if bias:
      b = 2*(t.rand(out_features) - 0.5) / t.sqrt(t.tensor(in_features))
      self.bias = t.nn.Parameter(b)
    else:
      self.bias = None

  def forward(self, x):
    '''
    x: shape (*, in_features)
    Return: shape (*, out_features)
    '''
    sum = einops.einsum(x, self.weight, \
                        '... in_features, out_features in_features -> \
                          ... out_features')
    if self.bias is not None:
      sum += self.bias
    return sum

  def extra_repr(self):
    return f"in_features={self.in_features}, \
     out_features={self.out_features}, bias={self.bias is not None}"

```


```python
class CustomFlatten(t.nn.Module):
  def __init__(self, start_dim=1, end_dim=-1):
    super().__init__()
    self.start_dim = start_dim
    self.end_dim = end_dim

  def forward(self, input):
    '''
    Flatten out dimensions from start_dim to end_dim, inclusive of both.
    '''
    shape = t.tensor(input.shape)
    end_dim = self.end_dim if self.end_dim >=0 else len(shape) + self.end_dim

    flattened_dim = t.prod(shape[self.start_dim:end_dim+1])
    flattened_shape = list(input.shape[:self.start_dim]) + \
      [int(flattened_dim)] + list(input.shape[end_dim+1:])
    return t.reshape(input, flattened_shape)

  def extra_repr(self):
    return ", ".join([f"{key}={getattr(self, key)}" for key in \
     ["start_dim", "end_dim"]])
```


```python
class CustomSequential(t.nn.Module):
  def __init__(self, *modules):
    super().__init__()
    for index, mod in enumerate(modules):
      self._modules[str(index)] = mod

  def __getitem__(self, index):
    index %= len(self._modules) # deal with negative indices
    return self._modules[str(index)]

  def __setitem__(self, index, module):
    index %= len(self._modules) # deal with negative indices
    self._modules[str(index)] = module

  def forward(self, x):
    '''
    Chain each module together, with the output from one feeding into
    the next one.
    '''
    for mod in self._modules.values():
      x = mod(x)
    return x
```


```python
class CustomBatchNorm2d(t.nn.Module):
  def __init__(self, num_features, eps=1e-05, momentum=0.1):
    '''
    Like nn.BatchNorm2d with track_running_stats=True and affine=True.

    Name the learnable affine parameters `weight` and `bias` in that order.
    '''
    super().__init__()
    self.weight = t.nn.Parameter(t.ones(num_features))
    self.bias = t.nn.Parameter(t.zeros(num_features))
    self.momentum = momentum
    self.eps = eps
    self.num_features = num_features

    self.register_buffer('running_mean', t.zeros(num_features))
    self.register_buffer('running_var', t.ones(num_features))
    self.register_buffer('num_batches_tracked', t.tensor(0))

  def forward(self, x):
    '''
    Normalize each channel.

    x: shape (batch, channels, height, width)
    Return: shape (batch, channels, height, width)
    '''
    if self.training:
      mean = t.mean(x, dim=(0, 2, 3), keepdim=True)
      var = t.var(x, unbiased=False, dim=(0, 2, 3), keepdim=True)
      self.running_mean = \
        (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
      self.running_var = \
        (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
      self.num_batches_tracked += 1
    else:
      mean = einops.rearrange(self.running_mean, 'c -> 1 c 1 1')
      var = einops.rearrange(self.running_var, 'c -> 1 c 1 1')

    weight = einops.rearrange(self.weight, 'c -> 1 c 1 1')
    bias = einops.rearrange(self.bias, 'c -> 1 c 1 1')
    return ((x - mean) / (var + self.eps).sqrt()) * weight + bias

  def extra_repr(self):
    keys = ['num_features', 'eps', 'momentum']
    return ", ".join([f'{key}={getattr(self, key)}' for key in keys])
```

For Conv2d and MaxPool2d, we'll cheat a little and use the preimplemented PyTorch `nn.functional`s for now. Implementing these further from scratch will be covered in the next section.


```python
def CustomConv2dFactory(conv2d):
  class CustomConv2d(t.nn.Module):
    def __init__(
      self, in_channels, out_channels, kernel_size, stride=1, padding=0, \
      bias=False):
      '''
      Same as torch.nn.Conv2d with bias=False.

      Name your weight field `self.weight` for compatibility with the PyTorch version.
      '''
      super().__init__()
      self.in_channels = in_channels
      self.out_channels = out_channels
      self.kernel_size = kernel_size
      self.stride = stride
      self.padding = padding
      sqrt_in = t. sqrt(t.tensor(in_channels * kernel_size**2))
      weight = ((2 * t.rand(out_channels, in_channels, \
                            kernel_size, kernel_size)) - 1) / sqrt_in
      self.weight = t.nn.Parameter(weight)
      self.stride = stride
      self.padding = padding

    def forward(self, x):
      return conv2d(x, self.weight, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
      keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
      return ", ".join([f"{key}={getattr(self, key)}" for key in keys])

  return CustomConv2d
```


```python
def CustomMaxPool2dFactory(maxpool2d):
  class CustomMaxPool2d(t.nn.Module):
    def __init__(self, kernel_size: int, stride = None, padding: int = 1):
      super().__init__()
      self.kernel_size = kernel_size
      self.stride = stride
      self.padding = padding

    def forward(self, x):
      return maxpool2d(x,self.kernel_size, self.stride, padding=self.padding)

    def extra_repr(self):
      keys = ["kernel_size", "stride", "padding"]
      return ", ".join([f"{key}={getattr(self,key)}" for key in keys])

  return CustomMaxPool2d
```


```python
CustomConv2d = CustomConv2dFactory(t.nn.functional.conv2d)
CustomMaxPool2d = CustomMaxPool2dFactory(t.nn.functional.max_pool2d)
```

### Construct and test ResNet50 using these building blocks


```python
Conv2dLayer = Conv2dFactory(CustomConv2d, CustomBatchNorm2d, CustomReLU)
BottleneckConvBlock = ConvBlockFactory(Conv2dLayer, CustomSequential, bottleneck=True)
ResidualBlock = ResidualBlockFactory(BottleneckConvBlock, Conv2dLayer, CustomReLU)
BlockGroup = BlockGroupFactory(ResidualBlock, CustomSequential)
ResNet = ResNetFactory(Conv2dLayer, CustomMaxPool2d, BlockGroup, AveragePool, \
                       CustomFlatten, CustomLinear, CustomSequential)

my_resnet50 = ResNet(n_blocks_per_group=[3, 4, 6, 3],
                     middle_features_per_group=[64, 128, 256, 512],
                     out_features_per_group=[256, 512, 1024, 2048])
```


```python
my_resnet50 = copy_weights(my_resnet50, pretrained_resnet50)
print_param_count(my_resnet50, pretrained_resnet50)
```

    Model 1, total params = 25557032
    Model 2, total params = 25557032
    All parameter counts match!



<style type="text/css">
#T_3c8f4_row0_col2, #T_3c8f4_row0_col3 {
  background-color: #238a8d;
  color: #f1f1f1;
}
#T_3c8f4_row1_col2, #T_3c8f4_row1_col3, #T_3c8f4_row2_col2, #T_3c8f4_row2_col3, #T_3c8f4_row4_col2, #T_3c8f4_row4_col3, #T_3c8f4_row5_col2, #T_3c8f4_row5_col3, #T_3c8f4_row7_col2, #T_3c8f4_row7_col3, #T_3c8f4_row8_col2, #T_3c8f4_row8_col3, #T_3c8f4_row16_col2, #T_3c8f4_row16_col3, #T_3c8f4_row17_col2, #T_3c8f4_row17_col3, #T_3c8f4_row19_col2, #T_3c8f4_row19_col3, #T_3c8f4_row20_col2, #T_3c8f4_row20_col3, #T_3c8f4_row25_col2, #T_3c8f4_row25_col3, #T_3c8f4_row26_col2, #T_3c8f4_row26_col3, #T_3c8f4_row28_col2, #T_3c8f4_row28_col3, #T_3c8f4_row29_col2, #T_3c8f4_row29_col3 {
  background-color: #440154;
  color: #f1f1f1;
}
#T_3c8f4_row3_col2, #T_3c8f4_row3_col3 {
  background-color: #2a778e;
  color: #f1f1f1;
}
#T_3c8f4_row6_col2, #T_3c8f4_row6_col3, #T_3c8f4_row18_col2, #T_3c8f4_row18_col3, #T_3c8f4_row27_col2, #T_3c8f4_row27_col3 {
  background-color: #23a983;
  color: #f1f1f1;
}
#T_3c8f4_row9_col2, #T_3c8f4_row9_col3, #T_3c8f4_row12_col2, #T_3c8f4_row12_col3, #T_3c8f4_row15_col2, #T_3c8f4_row15_col3, #T_3c8f4_row21_col2, #T_3c8f4_row21_col3, #T_3c8f4_row24_col2, #T_3c8f4_row24_col3, #T_3c8f4_row30_col2, #T_3c8f4_row30_col3 {
  background-color: #1f978b;
  color: #f1f1f1;
}
#T_3c8f4_row10_col2, #T_3c8f4_row10_col3, #T_3c8f4_row11_col2, #T_3c8f4_row11_col3, #T_3c8f4_row13_col2, #T_3c8f4_row13_col3, #T_3c8f4_row14_col2, #T_3c8f4_row14_col3, #T_3c8f4_row22_col2, #T_3c8f4_row22_col3, #T_3c8f4_row23_col2, #T_3c8f4_row23_col3, #T_3c8f4_row31_col2, #T_3c8f4_row31_col3, #T_3c8f4_row32_col2, #T_3c8f4_row32_col3, #T_3c8f4_row73_col2, #T_3c8f4_row73_col3, #T_3c8f4_row74_col2, #T_3c8f4_row74_col3, #T_3c8f4_row76_col2, #T_3c8f4_row76_col3, #T_3c8f4_row77_col2, #T_3c8f4_row77_col3, #T_3c8f4_row85_col2, #T_3c8f4_row85_col3, #T_3c8f4_row86_col2, #T_3c8f4_row86_col3, #T_3c8f4_row88_col2, #T_3c8f4_row88_col3, #T_3c8f4_row89_col2, #T_3c8f4_row89_col3, #T_3c8f4_row94_col2, #T_3c8f4_row94_col3, #T_3c8f4_row95_col2, #T_3c8f4_row95_col3, #T_3c8f4_row97_col2, #T_3c8f4_row97_col3, #T_3c8f4_row98_col2, #T_3c8f4_row98_col3, #T_3c8f4_row103_col2, #T_3c8f4_row103_col3, #T_3c8f4_row104_col2, #T_3c8f4_row104_col3, #T_3c8f4_row106_col2, #T_3c8f4_row106_col3, #T_3c8f4_row107_col2, #T_3c8f4_row107_col3, #T_3c8f4_row112_col2, #T_3c8f4_row112_col3, #T_3c8f4_row113_col2, #T_3c8f4_row113_col3, #T_3c8f4_row115_col2, #T_3c8f4_row115_col3, #T_3c8f4_row116_col2, #T_3c8f4_row116_col3, #T_3c8f4_row121_col2, #T_3c8f4_row121_col3, #T_3c8f4_row122_col2, #T_3c8f4_row122_col3, #T_3c8f4_row124_col2, #T_3c8f4_row124_col3, #T_3c8f4_row125_col2, #T_3c8f4_row125_col3 {
  background-color: #472e7c;
  color: #f1f1f1;
}
#T_3c8f4_row33_col2, #T_3c8f4_row33_col3 {
  background-color: #21a685;
  color: #f1f1f1;
}
#T_3c8f4_row34_col2, #T_3c8f4_row34_col3, #T_3c8f4_row35_col2, #T_3c8f4_row35_col3, #T_3c8f4_row37_col2, #T_3c8f4_row37_col3, #T_3c8f4_row38_col2, #T_3c8f4_row38_col3, #T_3c8f4_row46_col2, #T_3c8f4_row46_col3, #T_3c8f4_row47_col2, #T_3c8f4_row47_col3, #T_3c8f4_row49_col2, #T_3c8f4_row49_col3, #T_3c8f4_row50_col2, #T_3c8f4_row50_col3, #T_3c8f4_row55_col2, #T_3c8f4_row55_col3, #T_3c8f4_row56_col2, #T_3c8f4_row56_col3, #T_3c8f4_row58_col2, #T_3c8f4_row58_col3, #T_3c8f4_row59_col2, #T_3c8f4_row59_col3, #T_3c8f4_row64_col2, #T_3c8f4_row64_col3, #T_3c8f4_row65_col2, #T_3c8f4_row65_col3, #T_3c8f4_row67_col2, #T_3c8f4_row67_col3, #T_3c8f4_row68_col2, #T_3c8f4_row68_col3 {
  background-color: #48186a;
  color: #f1f1f1;
}
#T_3c8f4_row36_col2, #T_3c8f4_row36_col3, #T_3c8f4_row48_col2, #T_3c8f4_row48_col3, #T_3c8f4_row57_col2, #T_3c8f4_row57_col3, #T_3c8f4_row66_col2, #T_3c8f4_row66_col3 {
  background-color: #56c667;
  color: #000000;
}
#T_3c8f4_row39_col2, #T_3c8f4_row39_col3, #T_3c8f4_row45_col2, #T_3c8f4_row45_col3, #T_3c8f4_row51_col2, #T_3c8f4_row51_col3, #T_3c8f4_row54_col2, #T_3c8f4_row54_col3, #T_3c8f4_row60_col2, #T_3c8f4_row60_col3, #T_3c8f4_row63_col2, #T_3c8f4_row63_col3, #T_3c8f4_row69_col2, #T_3c8f4_row69_col3 {
  background-color: #32b67a;
  color: #f1f1f1;
}
#T_3c8f4_row40_col2, #T_3c8f4_row40_col3, #T_3c8f4_row41_col2, #T_3c8f4_row41_col3, #T_3c8f4_row43_col2, #T_3c8f4_row43_col3, #T_3c8f4_row44_col2, #T_3c8f4_row44_col3, #T_3c8f4_row52_col2, #T_3c8f4_row52_col3, #T_3c8f4_row53_col2, #T_3c8f4_row53_col3, #T_3c8f4_row61_col2, #T_3c8f4_row61_col3, #T_3c8f4_row62_col2, #T_3c8f4_row62_col3, #T_3c8f4_row70_col2, #T_3c8f4_row70_col3, #T_3c8f4_row71_col2, #T_3c8f4_row71_col3, #T_3c8f4_row130_col2, #T_3c8f4_row130_col3, #T_3c8f4_row131_col2, #T_3c8f4_row131_col3, #T_3c8f4_row133_col2, #T_3c8f4_row133_col3, #T_3c8f4_row134_col2, #T_3c8f4_row134_col3, #T_3c8f4_row142_col2, #T_3c8f4_row142_col3, #T_3c8f4_row143_col2, #T_3c8f4_row143_col3, #T_3c8f4_row145_col2, #T_3c8f4_row145_col3, #T_3c8f4_row146_col2, #T_3c8f4_row146_col3, #T_3c8f4_row151_col2, #T_3c8f4_row151_col3, #T_3c8f4_row152_col2, #T_3c8f4_row152_col3, #T_3c8f4_row154_col2, #T_3c8f4_row154_col3, #T_3c8f4_row155_col2, #T_3c8f4_row155_col3 {
  background-color: #414287;
  color: #f1f1f1;
}
#T_3c8f4_row42_col2, #T_3c8f4_row42_col3, #T_3c8f4_row72_col2, #T_3c8f4_row72_col3 {
  background-color: #50c46a;
  color: #000000;
}
#T_3c8f4_row75_col2, #T_3c8f4_row75_col3, #T_3c8f4_row87_col2, #T_3c8f4_row87_col3, #T_3c8f4_row96_col2, #T_3c8f4_row96_col3, #T_3c8f4_row105_col2, #T_3c8f4_row105_col3, #T_3c8f4_row114_col2, #T_3c8f4_row114_col3, #T_3c8f4_row123_col2, #T_3c8f4_row123_col3 {
  background-color: #a8db34;
  color: #000000;
}
#T_3c8f4_row78_col2, #T_3c8f4_row78_col3, #T_3c8f4_row84_col2, #T_3c8f4_row84_col3, #T_3c8f4_row90_col2, #T_3c8f4_row90_col3, #T_3c8f4_row93_col2, #T_3c8f4_row93_col3, #T_3c8f4_row99_col2, #T_3c8f4_row99_col3, #T_3c8f4_row102_col2, #T_3c8f4_row102_col3, #T_3c8f4_row108_col2, #T_3c8f4_row108_col3, #T_3c8f4_row111_col2, #T_3c8f4_row111_col3, #T_3c8f4_row117_col2, #T_3c8f4_row117_col3, #T_3c8f4_row120_col2, #T_3c8f4_row120_col3, #T_3c8f4_row126_col2, #T_3c8f4_row126_col3 {
  background-color: #75d054;
  color: #000000;
}
#T_3c8f4_row79_col2, #T_3c8f4_row79_col3, #T_3c8f4_row80_col2, #T_3c8f4_row80_col3, #T_3c8f4_row82_col2, #T_3c8f4_row82_col3, #T_3c8f4_row83_col2, #T_3c8f4_row83_col3, #T_3c8f4_row91_col2, #T_3c8f4_row91_col3, #T_3c8f4_row92_col2, #T_3c8f4_row92_col3, #T_3c8f4_row100_col2, #T_3c8f4_row100_col3, #T_3c8f4_row101_col2, #T_3c8f4_row101_col3, #T_3c8f4_row109_col2, #T_3c8f4_row109_col3, #T_3c8f4_row110_col2, #T_3c8f4_row110_col3, #T_3c8f4_row118_col2, #T_3c8f4_row118_col3, #T_3c8f4_row119_col2, #T_3c8f4_row119_col3, #T_3c8f4_row127_col2, #T_3c8f4_row127_col3, #T_3c8f4_row128_col2, #T_3c8f4_row128_col3 {
  background-color: #39558c;
  color: #f1f1f1;
}
#T_3c8f4_row81_col2, #T_3c8f4_row81_col3, #T_3c8f4_row129_col2, #T_3c8f4_row129_col3 {
  background-color: #a0da39;
  color: #000000;
}
#T_3c8f4_row132_col2, #T_3c8f4_row132_col3, #T_3c8f4_row144_col2, #T_3c8f4_row144_col3, #T_3c8f4_row153_col2, #T_3c8f4_row153_col3 {
  background-color: #fde725;
  color: #000000;
}
#T_3c8f4_row135_col2, #T_3c8f4_row135_col3, #T_3c8f4_row141_col2, #T_3c8f4_row141_col3, #T_3c8f4_row147_col2, #T_3c8f4_row147_col3, #T_3c8f4_row150_col2, #T_3c8f4_row150_col3, #T_3c8f4_row156_col2, #T_3c8f4_row156_col3 {
  background-color: #cde11d;
  color: #000000;
}
#T_3c8f4_row136_col2, #T_3c8f4_row136_col3, #T_3c8f4_row137_col2, #T_3c8f4_row137_col3, #T_3c8f4_row139_col2, #T_3c8f4_row139_col3, #T_3c8f4_row140_col2, #T_3c8f4_row140_col3, #T_3c8f4_row148_col2, #T_3c8f4_row148_col3, #T_3c8f4_row149_col2, #T_3c8f4_row149_col3, #T_3c8f4_row157_col2, #T_3c8f4_row157_col3, #T_3c8f4_row158_col2, #T_3c8f4_row158_col3 {
  background-color: #31678e;
  color: #f1f1f1;
}
#T_3c8f4_row138_col2, #T_3c8f4_row138_col3 {
  background-color: #f8e621;
  color: #000000;
}
#T_3c8f4_row159_col2, #T_3c8f4_row159_col3 {
  background-color: #f6e620;
  color: #000000;
}
#T_3c8f4_row160_col2, #T_3c8f4_row160_col3 {
  background-color: #3a548c;
  color: #f1f1f1;
}
</style>
<table id="T_3c8f4" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_3c8f4_level0_col0" class="col_heading level0 col0" >name_1</th>
      <th id="T_3c8f4_level0_col1" class="col_heading level0 col1" >shape_1</th>
      <th id="T_3c8f4_level0_col2" class="col_heading level0 col2" >num_params_1</th>
      <th id="T_3c8f4_level0_col3" class="col_heading level0 col3" >num_params_2</th>
      <th id="T_3c8f4_level0_col4" class="col_heading level0 col4" >shape_2</th>
      <th id="T_3c8f4_level0_col5" class="col_heading level0 col5" >name_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_3c8f4_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_3c8f4_row0_col0" class="data row0 col0" >input_layers.0.conv.weight</td>
      <td id="T_3c8f4_row0_col1" class="data row0 col1" >(64, 3, 7, 7)</td>
      <td id="T_3c8f4_row0_col2" class="data row0 col2" >9408</td>
      <td id="T_3c8f4_row0_col3" class="data row0 col3" >9408</td>
      <td id="T_3c8f4_row0_col4" class="data row0 col4" >(64, 3, 7, 7)</td>
      <td id="T_3c8f4_row0_col5" class="data row0 col5" >conv1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_3c8f4_row1_col0" class="data row1 col0" >input_layers.0.batchnorm2d.weight</td>
      <td id="T_3c8f4_row1_col1" class="data row1 col1" >(64,)</td>
      <td id="T_3c8f4_row1_col2" class="data row1 col2" >64</td>
      <td id="T_3c8f4_row1_col3" class="data row1 col3" >64</td>
      <td id="T_3c8f4_row1_col4" class="data row1 col4" >(64,)</td>
      <td id="T_3c8f4_row1_col5" class="data row1 col5" >bn1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_3c8f4_row2_col0" class="data row2 col0" >input_layers.0.batchnorm2d.bias</td>
      <td id="T_3c8f4_row2_col1" class="data row2 col1" >(64,)</td>
      <td id="T_3c8f4_row2_col2" class="data row2 col2" >64</td>
      <td id="T_3c8f4_row2_col3" class="data row2 col3" >64</td>
      <td id="T_3c8f4_row2_col4" class="data row2 col4" >(64,)</td>
      <td id="T_3c8f4_row2_col5" class="data row2 col5" >bn1.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_3c8f4_row3_col0" class="data row3 col0" >block_groups.0.block_group.0.left.conv_block.0.conv.weight</td>
      <td id="T_3c8f4_row3_col1" class="data row3 col1" >(64, 64, 1, 1)</td>
      <td id="T_3c8f4_row3_col2" class="data row3 col2" >4096</td>
      <td id="T_3c8f4_row3_col3" class="data row3 col3" >4096</td>
      <td id="T_3c8f4_row3_col4" class="data row3 col4" >(64, 64, 1, 1)</td>
      <td id="T_3c8f4_row3_col5" class="data row3 col5" >layer1.0.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_3c8f4_row4_col0" class="data row4 col0" >block_groups.0.block_group.0.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_3c8f4_row4_col1" class="data row4 col1" >(64,)</td>
      <td id="T_3c8f4_row4_col2" class="data row4 col2" >64</td>
      <td id="T_3c8f4_row4_col3" class="data row4 col3" >64</td>
      <td id="T_3c8f4_row4_col4" class="data row4 col4" >(64,)</td>
      <td id="T_3c8f4_row4_col5" class="data row4 col5" >layer1.0.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_3c8f4_row5_col0" class="data row5 col0" >block_groups.0.block_group.0.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_3c8f4_row5_col1" class="data row5 col1" >(64,)</td>
      <td id="T_3c8f4_row5_col2" class="data row5 col2" >64</td>
      <td id="T_3c8f4_row5_col3" class="data row5 col3" >64</td>
      <td id="T_3c8f4_row5_col4" class="data row5 col4" >(64,)</td>
      <td id="T_3c8f4_row5_col5" class="data row5 col5" >layer1.0.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_3c8f4_row6_col0" class="data row6 col0" >block_groups.0.block_group.0.left.conv_block.1.conv.weight</td>
      <td id="T_3c8f4_row6_col1" class="data row6 col1" >(64, 64, 3, 3)</td>
      <td id="T_3c8f4_row6_col2" class="data row6 col2" >36864</td>
      <td id="T_3c8f4_row6_col3" class="data row6 col3" >36864</td>
      <td id="T_3c8f4_row6_col4" class="data row6 col4" >(64, 64, 3, 3)</td>
      <td id="T_3c8f4_row6_col5" class="data row6 col5" >layer1.0.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_3c8f4_row7_col0" class="data row7 col0" >block_groups.0.block_group.0.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_3c8f4_row7_col1" class="data row7 col1" >(64,)</td>
      <td id="T_3c8f4_row7_col2" class="data row7 col2" >64</td>
      <td id="T_3c8f4_row7_col3" class="data row7 col3" >64</td>
      <td id="T_3c8f4_row7_col4" class="data row7 col4" >(64,)</td>
      <td id="T_3c8f4_row7_col5" class="data row7 col5" >layer1.0.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_3c8f4_row8_col0" class="data row8 col0" >block_groups.0.block_group.0.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_3c8f4_row8_col1" class="data row8 col1" >(64,)</td>
      <td id="T_3c8f4_row8_col2" class="data row8 col2" >64</td>
      <td id="T_3c8f4_row8_col3" class="data row8 col3" >64</td>
      <td id="T_3c8f4_row8_col4" class="data row8 col4" >(64,)</td>
      <td id="T_3c8f4_row8_col5" class="data row8 col5" >layer1.0.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_3c8f4_row9_col0" class="data row9 col0" >block_groups.0.block_group.0.left.conv_block.2.conv.weight</td>
      <td id="T_3c8f4_row9_col1" class="data row9 col1" >(256, 64, 1, 1)</td>
      <td id="T_3c8f4_row9_col2" class="data row9 col2" >16384</td>
      <td id="T_3c8f4_row9_col3" class="data row9 col3" >16384</td>
      <td id="T_3c8f4_row9_col4" class="data row9 col4" >(256, 64, 1, 1)</td>
      <td id="T_3c8f4_row9_col5" class="data row9 col5" >layer1.0.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_3c8f4_row10_col0" class="data row10 col0" >block_groups.0.block_group.0.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_3c8f4_row10_col1" class="data row10 col1" >(256,)</td>
      <td id="T_3c8f4_row10_col2" class="data row10 col2" >256</td>
      <td id="T_3c8f4_row10_col3" class="data row10 col3" >256</td>
      <td id="T_3c8f4_row10_col4" class="data row10 col4" >(256,)</td>
      <td id="T_3c8f4_row10_col5" class="data row10 col5" >layer1.0.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_3c8f4_row11_col0" class="data row11 col0" >block_groups.0.block_group.0.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_3c8f4_row11_col1" class="data row11 col1" >(256,)</td>
      <td id="T_3c8f4_row11_col2" class="data row11 col2" >256</td>
      <td id="T_3c8f4_row11_col3" class="data row11 col3" >256</td>
      <td id="T_3c8f4_row11_col4" class="data row11 col4" >(256,)</td>
      <td id="T_3c8f4_row11_col5" class="data row11 col5" >layer1.0.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_3c8f4_row12_col0" class="data row12 col0" >block_groups.0.block_group.0.right.conv.weight</td>
      <td id="T_3c8f4_row12_col1" class="data row12 col1" >(256, 64, 1, 1)</td>
      <td id="T_3c8f4_row12_col2" class="data row12 col2" >16384</td>
      <td id="T_3c8f4_row12_col3" class="data row12 col3" >16384</td>
      <td id="T_3c8f4_row12_col4" class="data row12 col4" >(256, 64, 1, 1)</td>
      <td id="T_3c8f4_row12_col5" class="data row12 col5" >layer1.0.downsample.0.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_3c8f4_row13_col0" class="data row13 col0" >block_groups.0.block_group.0.right.batchnorm2d.weight</td>
      <td id="T_3c8f4_row13_col1" class="data row13 col1" >(256,)</td>
      <td id="T_3c8f4_row13_col2" class="data row13 col2" >256</td>
      <td id="T_3c8f4_row13_col3" class="data row13 col3" >256</td>
      <td id="T_3c8f4_row13_col4" class="data row13 col4" >(256,)</td>
      <td id="T_3c8f4_row13_col5" class="data row13 col5" >layer1.0.downsample.1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_3c8f4_row14_col0" class="data row14 col0" >block_groups.0.block_group.0.right.batchnorm2d.bias</td>
      <td id="T_3c8f4_row14_col1" class="data row14 col1" >(256,)</td>
      <td id="T_3c8f4_row14_col2" class="data row14 col2" >256</td>
      <td id="T_3c8f4_row14_col3" class="data row14 col3" >256</td>
      <td id="T_3c8f4_row14_col4" class="data row14 col4" >(256,)</td>
      <td id="T_3c8f4_row14_col5" class="data row14 col5" >layer1.0.downsample.1.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_3c8f4_row15_col0" class="data row15 col0" >block_groups.0.block_group.1.left.conv_block.0.conv.weight</td>
      <td id="T_3c8f4_row15_col1" class="data row15 col1" >(64, 256, 1, 1)</td>
      <td id="T_3c8f4_row15_col2" class="data row15 col2" >16384</td>
      <td id="T_3c8f4_row15_col3" class="data row15 col3" >16384</td>
      <td id="T_3c8f4_row15_col4" class="data row15 col4" >(64, 256, 1, 1)</td>
      <td id="T_3c8f4_row15_col5" class="data row15 col5" >layer1.1.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_3c8f4_row16_col0" class="data row16 col0" >block_groups.0.block_group.1.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_3c8f4_row16_col1" class="data row16 col1" >(64,)</td>
      <td id="T_3c8f4_row16_col2" class="data row16 col2" >64</td>
      <td id="T_3c8f4_row16_col3" class="data row16 col3" >64</td>
      <td id="T_3c8f4_row16_col4" class="data row16 col4" >(64,)</td>
      <td id="T_3c8f4_row16_col5" class="data row16 col5" >layer1.1.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_3c8f4_row17_col0" class="data row17 col0" >block_groups.0.block_group.1.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_3c8f4_row17_col1" class="data row17 col1" >(64,)</td>
      <td id="T_3c8f4_row17_col2" class="data row17 col2" >64</td>
      <td id="T_3c8f4_row17_col3" class="data row17 col3" >64</td>
      <td id="T_3c8f4_row17_col4" class="data row17 col4" >(64,)</td>
      <td id="T_3c8f4_row17_col5" class="data row17 col5" >layer1.1.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_3c8f4_row18_col0" class="data row18 col0" >block_groups.0.block_group.1.left.conv_block.1.conv.weight</td>
      <td id="T_3c8f4_row18_col1" class="data row18 col1" >(64, 64, 3, 3)</td>
      <td id="T_3c8f4_row18_col2" class="data row18 col2" >36864</td>
      <td id="T_3c8f4_row18_col3" class="data row18 col3" >36864</td>
      <td id="T_3c8f4_row18_col4" class="data row18 col4" >(64, 64, 3, 3)</td>
      <td id="T_3c8f4_row18_col5" class="data row18 col5" >layer1.1.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_3c8f4_row19_col0" class="data row19 col0" >block_groups.0.block_group.1.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_3c8f4_row19_col1" class="data row19 col1" >(64,)</td>
      <td id="T_3c8f4_row19_col2" class="data row19 col2" >64</td>
      <td id="T_3c8f4_row19_col3" class="data row19 col3" >64</td>
      <td id="T_3c8f4_row19_col4" class="data row19 col4" >(64,)</td>
      <td id="T_3c8f4_row19_col5" class="data row19 col5" >layer1.1.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row20" class="row_heading level0 row20" >20</th>
      <td id="T_3c8f4_row20_col0" class="data row20 col0" >block_groups.0.block_group.1.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_3c8f4_row20_col1" class="data row20 col1" >(64,)</td>
      <td id="T_3c8f4_row20_col2" class="data row20 col2" >64</td>
      <td id="T_3c8f4_row20_col3" class="data row20 col3" >64</td>
      <td id="T_3c8f4_row20_col4" class="data row20 col4" >(64,)</td>
      <td id="T_3c8f4_row20_col5" class="data row20 col5" >layer1.1.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row21" class="row_heading level0 row21" >21</th>
      <td id="T_3c8f4_row21_col0" class="data row21 col0" >block_groups.0.block_group.1.left.conv_block.2.conv.weight</td>
      <td id="T_3c8f4_row21_col1" class="data row21 col1" >(256, 64, 1, 1)</td>
      <td id="T_3c8f4_row21_col2" class="data row21 col2" >16384</td>
      <td id="T_3c8f4_row21_col3" class="data row21 col3" >16384</td>
      <td id="T_3c8f4_row21_col4" class="data row21 col4" >(256, 64, 1, 1)</td>
      <td id="T_3c8f4_row21_col5" class="data row21 col5" >layer1.1.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row22" class="row_heading level0 row22" >22</th>
      <td id="T_3c8f4_row22_col0" class="data row22 col0" >block_groups.0.block_group.1.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_3c8f4_row22_col1" class="data row22 col1" >(256,)</td>
      <td id="T_3c8f4_row22_col2" class="data row22 col2" >256</td>
      <td id="T_3c8f4_row22_col3" class="data row22 col3" >256</td>
      <td id="T_3c8f4_row22_col4" class="data row22 col4" >(256,)</td>
      <td id="T_3c8f4_row22_col5" class="data row22 col5" >layer1.1.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row23" class="row_heading level0 row23" >23</th>
      <td id="T_3c8f4_row23_col0" class="data row23 col0" >block_groups.0.block_group.1.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_3c8f4_row23_col1" class="data row23 col1" >(256,)</td>
      <td id="T_3c8f4_row23_col2" class="data row23 col2" >256</td>
      <td id="T_3c8f4_row23_col3" class="data row23 col3" >256</td>
      <td id="T_3c8f4_row23_col4" class="data row23 col4" >(256,)</td>
      <td id="T_3c8f4_row23_col5" class="data row23 col5" >layer1.1.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row24" class="row_heading level0 row24" >24</th>
      <td id="T_3c8f4_row24_col0" class="data row24 col0" >block_groups.0.block_group.2.left.conv_block.0.conv.weight</td>
      <td id="T_3c8f4_row24_col1" class="data row24 col1" >(64, 256, 1, 1)</td>
      <td id="T_3c8f4_row24_col2" class="data row24 col2" >16384</td>
      <td id="T_3c8f4_row24_col3" class="data row24 col3" >16384</td>
      <td id="T_3c8f4_row24_col4" class="data row24 col4" >(64, 256, 1, 1)</td>
      <td id="T_3c8f4_row24_col5" class="data row24 col5" >layer1.2.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row25" class="row_heading level0 row25" >25</th>
      <td id="T_3c8f4_row25_col0" class="data row25 col0" >block_groups.0.block_group.2.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_3c8f4_row25_col1" class="data row25 col1" >(64,)</td>
      <td id="T_3c8f4_row25_col2" class="data row25 col2" >64</td>
      <td id="T_3c8f4_row25_col3" class="data row25 col3" >64</td>
      <td id="T_3c8f4_row25_col4" class="data row25 col4" >(64,)</td>
      <td id="T_3c8f4_row25_col5" class="data row25 col5" >layer1.2.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row26" class="row_heading level0 row26" >26</th>
      <td id="T_3c8f4_row26_col0" class="data row26 col0" >block_groups.0.block_group.2.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_3c8f4_row26_col1" class="data row26 col1" >(64,)</td>
      <td id="T_3c8f4_row26_col2" class="data row26 col2" >64</td>
      <td id="T_3c8f4_row26_col3" class="data row26 col3" >64</td>
      <td id="T_3c8f4_row26_col4" class="data row26 col4" >(64,)</td>
      <td id="T_3c8f4_row26_col5" class="data row26 col5" >layer1.2.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row27" class="row_heading level0 row27" >27</th>
      <td id="T_3c8f4_row27_col0" class="data row27 col0" >block_groups.0.block_group.2.left.conv_block.1.conv.weight</td>
      <td id="T_3c8f4_row27_col1" class="data row27 col1" >(64, 64, 3, 3)</td>
      <td id="T_3c8f4_row27_col2" class="data row27 col2" >36864</td>
      <td id="T_3c8f4_row27_col3" class="data row27 col3" >36864</td>
      <td id="T_3c8f4_row27_col4" class="data row27 col4" >(64, 64, 3, 3)</td>
      <td id="T_3c8f4_row27_col5" class="data row27 col5" >layer1.2.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row28" class="row_heading level0 row28" >28</th>
      <td id="T_3c8f4_row28_col0" class="data row28 col0" >block_groups.0.block_group.2.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_3c8f4_row28_col1" class="data row28 col1" >(64,)</td>
      <td id="T_3c8f4_row28_col2" class="data row28 col2" >64</td>
      <td id="T_3c8f4_row28_col3" class="data row28 col3" >64</td>
      <td id="T_3c8f4_row28_col4" class="data row28 col4" >(64,)</td>
      <td id="T_3c8f4_row28_col5" class="data row28 col5" >layer1.2.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row29" class="row_heading level0 row29" >29</th>
      <td id="T_3c8f4_row29_col0" class="data row29 col0" >block_groups.0.block_group.2.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_3c8f4_row29_col1" class="data row29 col1" >(64,)</td>
      <td id="T_3c8f4_row29_col2" class="data row29 col2" >64</td>
      <td id="T_3c8f4_row29_col3" class="data row29 col3" >64</td>
      <td id="T_3c8f4_row29_col4" class="data row29 col4" >(64,)</td>
      <td id="T_3c8f4_row29_col5" class="data row29 col5" >layer1.2.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row30" class="row_heading level0 row30" >30</th>
      <td id="T_3c8f4_row30_col0" class="data row30 col0" >block_groups.0.block_group.2.left.conv_block.2.conv.weight</td>
      <td id="T_3c8f4_row30_col1" class="data row30 col1" >(256, 64, 1, 1)</td>
      <td id="T_3c8f4_row30_col2" class="data row30 col2" >16384</td>
      <td id="T_3c8f4_row30_col3" class="data row30 col3" >16384</td>
      <td id="T_3c8f4_row30_col4" class="data row30 col4" >(256, 64, 1, 1)</td>
      <td id="T_3c8f4_row30_col5" class="data row30 col5" >layer1.2.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row31" class="row_heading level0 row31" >31</th>
      <td id="T_3c8f4_row31_col0" class="data row31 col0" >block_groups.0.block_group.2.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_3c8f4_row31_col1" class="data row31 col1" >(256,)</td>
      <td id="T_3c8f4_row31_col2" class="data row31 col2" >256</td>
      <td id="T_3c8f4_row31_col3" class="data row31 col3" >256</td>
      <td id="T_3c8f4_row31_col4" class="data row31 col4" >(256,)</td>
      <td id="T_3c8f4_row31_col5" class="data row31 col5" >layer1.2.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row32" class="row_heading level0 row32" >32</th>
      <td id="T_3c8f4_row32_col0" class="data row32 col0" >block_groups.0.block_group.2.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_3c8f4_row32_col1" class="data row32 col1" >(256,)</td>
      <td id="T_3c8f4_row32_col2" class="data row32 col2" >256</td>
      <td id="T_3c8f4_row32_col3" class="data row32 col3" >256</td>
      <td id="T_3c8f4_row32_col4" class="data row32 col4" >(256,)</td>
      <td id="T_3c8f4_row32_col5" class="data row32 col5" >layer1.2.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row33" class="row_heading level0 row33" >33</th>
      <td id="T_3c8f4_row33_col0" class="data row33 col0" >block_groups.1.block_group.0.left.conv_block.0.conv.weight</td>
      <td id="T_3c8f4_row33_col1" class="data row33 col1" >(128, 256, 1, 1)</td>
      <td id="T_3c8f4_row33_col2" class="data row33 col2" >32768</td>
      <td id="T_3c8f4_row33_col3" class="data row33 col3" >32768</td>
      <td id="T_3c8f4_row33_col4" class="data row33 col4" >(128, 256, 1, 1)</td>
      <td id="T_3c8f4_row33_col5" class="data row33 col5" >layer2.0.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row34" class="row_heading level0 row34" >34</th>
      <td id="T_3c8f4_row34_col0" class="data row34 col0" >block_groups.1.block_group.0.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_3c8f4_row34_col1" class="data row34 col1" >(128,)</td>
      <td id="T_3c8f4_row34_col2" class="data row34 col2" >128</td>
      <td id="T_3c8f4_row34_col3" class="data row34 col3" >128</td>
      <td id="T_3c8f4_row34_col4" class="data row34 col4" >(128,)</td>
      <td id="T_3c8f4_row34_col5" class="data row34 col5" >layer2.0.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row35" class="row_heading level0 row35" >35</th>
      <td id="T_3c8f4_row35_col0" class="data row35 col0" >block_groups.1.block_group.0.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_3c8f4_row35_col1" class="data row35 col1" >(128,)</td>
      <td id="T_3c8f4_row35_col2" class="data row35 col2" >128</td>
      <td id="T_3c8f4_row35_col3" class="data row35 col3" >128</td>
      <td id="T_3c8f4_row35_col4" class="data row35 col4" >(128,)</td>
      <td id="T_3c8f4_row35_col5" class="data row35 col5" >layer2.0.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row36" class="row_heading level0 row36" >36</th>
      <td id="T_3c8f4_row36_col0" class="data row36 col0" >block_groups.1.block_group.0.left.conv_block.1.conv.weight</td>
      <td id="T_3c8f4_row36_col1" class="data row36 col1" >(128, 128, 3, 3)</td>
      <td id="T_3c8f4_row36_col2" class="data row36 col2" >147456</td>
      <td id="T_3c8f4_row36_col3" class="data row36 col3" >147456</td>
      <td id="T_3c8f4_row36_col4" class="data row36 col4" >(128, 128, 3, 3)</td>
      <td id="T_3c8f4_row36_col5" class="data row36 col5" >layer2.0.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row37" class="row_heading level0 row37" >37</th>
      <td id="T_3c8f4_row37_col0" class="data row37 col0" >block_groups.1.block_group.0.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_3c8f4_row37_col1" class="data row37 col1" >(128,)</td>
      <td id="T_3c8f4_row37_col2" class="data row37 col2" >128</td>
      <td id="T_3c8f4_row37_col3" class="data row37 col3" >128</td>
      <td id="T_3c8f4_row37_col4" class="data row37 col4" >(128,)</td>
      <td id="T_3c8f4_row37_col5" class="data row37 col5" >layer2.0.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row38" class="row_heading level0 row38" >38</th>
      <td id="T_3c8f4_row38_col0" class="data row38 col0" >block_groups.1.block_group.0.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_3c8f4_row38_col1" class="data row38 col1" >(128,)</td>
      <td id="T_3c8f4_row38_col2" class="data row38 col2" >128</td>
      <td id="T_3c8f4_row38_col3" class="data row38 col3" >128</td>
      <td id="T_3c8f4_row38_col4" class="data row38 col4" >(128,)</td>
      <td id="T_3c8f4_row38_col5" class="data row38 col5" >layer2.0.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row39" class="row_heading level0 row39" >39</th>
      <td id="T_3c8f4_row39_col0" class="data row39 col0" >block_groups.1.block_group.0.left.conv_block.2.conv.weight</td>
      <td id="T_3c8f4_row39_col1" class="data row39 col1" >(512, 128, 1, 1)</td>
      <td id="T_3c8f4_row39_col2" class="data row39 col2" >65536</td>
      <td id="T_3c8f4_row39_col3" class="data row39 col3" >65536</td>
      <td id="T_3c8f4_row39_col4" class="data row39 col4" >(512, 128, 1, 1)</td>
      <td id="T_3c8f4_row39_col5" class="data row39 col5" >layer2.0.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row40" class="row_heading level0 row40" >40</th>
      <td id="T_3c8f4_row40_col0" class="data row40 col0" >block_groups.1.block_group.0.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_3c8f4_row40_col1" class="data row40 col1" >(512,)</td>
      <td id="T_3c8f4_row40_col2" class="data row40 col2" >512</td>
      <td id="T_3c8f4_row40_col3" class="data row40 col3" >512</td>
      <td id="T_3c8f4_row40_col4" class="data row40 col4" >(512,)</td>
      <td id="T_3c8f4_row40_col5" class="data row40 col5" >layer2.0.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row41" class="row_heading level0 row41" >41</th>
      <td id="T_3c8f4_row41_col0" class="data row41 col0" >block_groups.1.block_group.0.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_3c8f4_row41_col1" class="data row41 col1" >(512,)</td>
      <td id="T_3c8f4_row41_col2" class="data row41 col2" >512</td>
      <td id="T_3c8f4_row41_col3" class="data row41 col3" >512</td>
      <td id="T_3c8f4_row41_col4" class="data row41 col4" >(512,)</td>
      <td id="T_3c8f4_row41_col5" class="data row41 col5" >layer2.0.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row42" class="row_heading level0 row42" >42</th>
      <td id="T_3c8f4_row42_col0" class="data row42 col0" >block_groups.1.block_group.0.right.conv.weight</td>
      <td id="T_3c8f4_row42_col1" class="data row42 col1" >(512, 256, 1, 1)</td>
      <td id="T_3c8f4_row42_col2" class="data row42 col2" >131072</td>
      <td id="T_3c8f4_row42_col3" class="data row42 col3" >131072</td>
      <td id="T_3c8f4_row42_col4" class="data row42 col4" >(512, 256, 1, 1)</td>
      <td id="T_3c8f4_row42_col5" class="data row42 col5" >layer2.0.downsample.0.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row43" class="row_heading level0 row43" >43</th>
      <td id="T_3c8f4_row43_col0" class="data row43 col0" >block_groups.1.block_group.0.right.batchnorm2d.weight</td>
      <td id="T_3c8f4_row43_col1" class="data row43 col1" >(512,)</td>
      <td id="T_3c8f4_row43_col2" class="data row43 col2" >512</td>
      <td id="T_3c8f4_row43_col3" class="data row43 col3" >512</td>
      <td id="T_3c8f4_row43_col4" class="data row43 col4" >(512,)</td>
      <td id="T_3c8f4_row43_col5" class="data row43 col5" >layer2.0.downsample.1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row44" class="row_heading level0 row44" >44</th>
      <td id="T_3c8f4_row44_col0" class="data row44 col0" >block_groups.1.block_group.0.right.batchnorm2d.bias</td>
      <td id="T_3c8f4_row44_col1" class="data row44 col1" >(512,)</td>
      <td id="T_3c8f4_row44_col2" class="data row44 col2" >512</td>
      <td id="T_3c8f4_row44_col3" class="data row44 col3" >512</td>
      <td id="T_3c8f4_row44_col4" class="data row44 col4" >(512,)</td>
      <td id="T_3c8f4_row44_col5" class="data row44 col5" >layer2.0.downsample.1.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row45" class="row_heading level0 row45" >45</th>
      <td id="T_3c8f4_row45_col0" class="data row45 col0" >block_groups.1.block_group.1.left.conv_block.0.conv.weight</td>
      <td id="T_3c8f4_row45_col1" class="data row45 col1" >(128, 512, 1, 1)</td>
      <td id="T_3c8f4_row45_col2" class="data row45 col2" >65536</td>
      <td id="T_3c8f4_row45_col3" class="data row45 col3" >65536</td>
      <td id="T_3c8f4_row45_col4" class="data row45 col4" >(128, 512, 1, 1)</td>
      <td id="T_3c8f4_row45_col5" class="data row45 col5" >layer2.1.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row46" class="row_heading level0 row46" >46</th>
      <td id="T_3c8f4_row46_col0" class="data row46 col0" >block_groups.1.block_group.1.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_3c8f4_row46_col1" class="data row46 col1" >(128,)</td>
      <td id="T_3c8f4_row46_col2" class="data row46 col2" >128</td>
      <td id="T_3c8f4_row46_col3" class="data row46 col3" >128</td>
      <td id="T_3c8f4_row46_col4" class="data row46 col4" >(128,)</td>
      <td id="T_3c8f4_row46_col5" class="data row46 col5" >layer2.1.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row47" class="row_heading level0 row47" >47</th>
      <td id="T_3c8f4_row47_col0" class="data row47 col0" >block_groups.1.block_group.1.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_3c8f4_row47_col1" class="data row47 col1" >(128,)</td>
      <td id="T_3c8f4_row47_col2" class="data row47 col2" >128</td>
      <td id="T_3c8f4_row47_col3" class="data row47 col3" >128</td>
      <td id="T_3c8f4_row47_col4" class="data row47 col4" >(128,)</td>
      <td id="T_3c8f4_row47_col5" class="data row47 col5" >layer2.1.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row48" class="row_heading level0 row48" >48</th>
      <td id="T_3c8f4_row48_col0" class="data row48 col0" >block_groups.1.block_group.1.left.conv_block.1.conv.weight</td>
      <td id="T_3c8f4_row48_col1" class="data row48 col1" >(128, 128, 3, 3)</td>
      <td id="T_3c8f4_row48_col2" class="data row48 col2" >147456</td>
      <td id="T_3c8f4_row48_col3" class="data row48 col3" >147456</td>
      <td id="T_3c8f4_row48_col4" class="data row48 col4" >(128, 128, 3, 3)</td>
      <td id="T_3c8f4_row48_col5" class="data row48 col5" >layer2.1.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row49" class="row_heading level0 row49" >49</th>
      <td id="T_3c8f4_row49_col0" class="data row49 col0" >block_groups.1.block_group.1.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_3c8f4_row49_col1" class="data row49 col1" >(128,)</td>
      <td id="T_3c8f4_row49_col2" class="data row49 col2" >128</td>
      <td id="T_3c8f4_row49_col3" class="data row49 col3" >128</td>
      <td id="T_3c8f4_row49_col4" class="data row49 col4" >(128,)</td>
      <td id="T_3c8f4_row49_col5" class="data row49 col5" >layer2.1.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row50" class="row_heading level0 row50" >50</th>
      <td id="T_3c8f4_row50_col0" class="data row50 col0" >block_groups.1.block_group.1.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_3c8f4_row50_col1" class="data row50 col1" >(128,)</td>
      <td id="T_3c8f4_row50_col2" class="data row50 col2" >128</td>
      <td id="T_3c8f4_row50_col3" class="data row50 col3" >128</td>
      <td id="T_3c8f4_row50_col4" class="data row50 col4" >(128,)</td>
      <td id="T_3c8f4_row50_col5" class="data row50 col5" >layer2.1.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row51" class="row_heading level0 row51" >51</th>
      <td id="T_3c8f4_row51_col0" class="data row51 col0" >block_groups.1.block_group.1.left.conv_block.2.conv.weight</td>
      <td id="T_3c8f4_row51_col1" class="data row51 col1" >(512, 128, 1, 1)</td>
      <td id="T_3c8f4_row51_col2" class="data row51 col2" >65536</td>
      <td id="T_3c8f4_row51_col3" class="data row51 col3" >65536</td>
      <td id="T_3c8f4_row51_col4" class="data row51 col4" >(512, 128, 1, 1)</td>
      <td id="T_3c8f4_row51_col5" class="data row51 col5" >layer2.1.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row52" class="row_heading level0 row52" >52</th>
      <td id="T_3c8f4_row52_col0" class="data row52 col0" >block_groups.1.block_group.1.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_3c8f4_row52_col1" class="data row52 col1" >(512,)</td>
      <td id="T_3c8f4_row52_col2" class="data row52 col2" >512</td>
      <td id="T_3c8f4_row52_col3" class="data row52 col3" >512</td>
      <td id="T_3c8f4_row52_col4" class="data row52 col4" >(512,)</td>
      <td id="T_3c8f4_row52_col5" class="data row52 col5" >layer2.1.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row53" class="row_heading level0 row53" >53</th>
      <td id="T_3c8f4_row53_col0" class="data row53 col0" >block_groups.1.block_group.1.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_3c8f4_row53_col1" class="data row53 col1" >(512,)</td>
      <td id="T_3c8f4_row53_col2" class="data row53 col2" >512</td>
      <td id="T_3c8f4_row53_col3" class="data row53 col3" >512</td>
      <td id="T_3c8f4_row53_col4" class="data row53 col4" >(512,)</td>
      <td id="T_3c8f4_row53_col5" class="data row53 col5" >layer2.1.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row54" class="row_heading level0 row54" >54</th>
      <td id="T_3c8f4_row54_col0" class="data row54 col0" >block_groups.1.block_group.2.left.conv_block.0.conv.weight</td>
      <td id="T_3c8f4_row54_col1" class="data row54 col1" >(128, 512, 1, 1)</td>
      <td id="T_3c8f4_row54_col2" class="data row54 col2" >65536</td>
      <td id="T_3c8f4_row54_col3" class="data row54 col3" >65536</td>
      <td id="T_3c8f4_row54_col4" class="data row54 col4" >(128, 512, 1, 1)</td>
      <td id="T_3c8f4_row54_col5" class="data row54 col5" >layer2.2.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row55" class="row_heading level0 row55" >55</th>
      <td id="T_3c8f4_row55_col0" class="data row55 col0" >block_groups.1.block_group.2.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_3c8f4_row55_col1" class="data row55 col1" >(128,)</td>
      <td id="T_3c8f4_row55_col2" class="data row55 col2" >128</td>
      <td id="T_3c8f4_row55_col3" class="data row55 col3" >128</td>
      <td id="T_3c8f4_row55_col4" class="data row55 col4" >(128,)</td>
      <td id="T_3c8f4_row55_col5" class="data row55 col5" >layer2.2.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row56" class="row_heading level0 row56" >56</th>
      <td id="T_3c8f4_row56_col0" class="data row56 col0" >block_groups.1.block_group.2.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_3c8f4_row56_col1" class="data row56 col1" >(128,)</td>
      <td id="T_3c8f4_row56_col2" class="data row56 col2" >128</td>
      <td id="T_3c8f4_row56_col3" class="data row56 col3" >128</td>
      <td id="T_3c8f4_row56_col4" class="data row56 col4" >(128,)</td>
      <td id="T_3c8f4_row56_col5" class="data row56 col5" >layer2.2.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row57" class="row_heading level0 row57" >57</th>
      <td id="T_3c8f4_row57_col0" class="data row57 col0" >block_groups.1.block_group.2.left.conv_block.1.conv.weight</td>
      <td id="T_3c8f4_row57_col1" class="data row57 col1" >(128, 128, 3, 3)</td>
      <td id="T_3c8f4_row57_col2" class="data row57 col2" >147456</td>
      <td id="T_3c8f4_row57_col3" class="data row57 col3" >147456</td>
      <td id="T_3c8f4_row57_col4" class="data row57 col4" >(128, 128, 3, 3)</td>
      <td id="T_3c8f4_row57_col5" class="data row57 col5" >layer2.2.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row58" class="row_heading level0 row58" >58</th>
      <td id="T_3c8f4_row58_col0" class="data row58 col0" >block_groups.1.block_group.2.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_3c8f4_row58_col1" class="data row58 col1" >(128,)</td>
      <td id="T_3c8f4_row58_col2" class="data row58 col2" >128</td>
      <td id="T_3c8f4_row58_col3" class="data row58 col3" >128</td>
      <td id="T_3c8f4_row58_col4" class="data row58 col4" >(128,)</td>
      <td id="T_3c8f4_row58_col5" class="data row58 col5" >layer2.2.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row59" class="row_heading level0 row59" >59</th>
      <td id="T_3c8f4_row59_col0" class="data row59 col0" >block_groups.1.block_group.2.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_3c8f4_row59_col1" class="data row59 col1" >(128,)</td>
      <td id="T_3c8f4_row59_col2" class="data row59 col2" >128</td>
      <td id="T_3c8f4_row59_col3" class="data row59 col3" >128</td>
      <td id="T_3c8f4_row59_col4" class="data row59 col4" >(128,)</td>
      <td id="T_3c8f4_row59_col5" class="data row59 col5" >layer2.2.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row60" class="row_heading level0 row60" >60</th>
      <td id="T_3c8f4_row60_col0" class="data row60 col0" >block_groups.1.block_group.2.left.conv_block.2.conv.weight</td>
      <td id="T_3c8f4_row60_col1" class="data row60 col1" >(512, 128, 1, 1)</td>
      <td id="T_3c8f4_row60_col2" class="data row60 col2" >65536</td>
      <td id="T_3c8f4_row60_col3" class="data row60 col3" >65536</td>
      <td id="T_3c8f4_row60_col4" class="data row60 col4" >(512, 128, 1, 1)</td>
      <td id="T_3c8f4_row60_col5" class="data row60 col5" >layer2.2.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row61" class="row_heading level0 row61" >61</th>
      <td id="T_3c8f4_row61_col0" class="data row61 col0" >block_groups.1.block_group.2.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_3c8f4_row61_col1" class="data row61 col1" >(512,)</td>
      <td id="T_3c8f4_row61_col2" class="data row61 col2" >512</td>
      <td id="T_3c8f4_row61_col3" class="data row61 col3" >512</td>
      <td id="T_3c8f4_row61_col4" class="data row61 col4" >(512,)</td>
      <td id="T_3c8f4_row61_col5" class="data row61 col5" >layer2.2.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row62" class="row_heading level0 row62" >62</th>
      <td id="T_3c8f4_row62_col0" class="data row62 col0" >block_groups.1.block_group.2.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_3c8f4_row62_col1" class="data row62 col1" >(512,)</td>
      <td id="T_3c8f4_row62_col2" class="data row62 col2" >512</td>
      <td id="T_3c8f4_row62_col3" class="data row62 col3" >512</td>
      <td id="T_3c8f4_row62_col4" class="data row62 col4" >(512,)</td>
      <td id="T_3c8f4_row62_col5" class="data row62 col5" >layer2.2.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row63" class="row_heading level0 row63" >63</th>
      <td id="T_3c8f4_row63_col0" class="data row63 col0" >block_groups.1.block_group.3.left.conv_block.0.conv.weight</td>
      <td id="T_3c8f4_row63_col1" class="data row63 col1" >(128, 512, 1, 1)</td>
      <td id="T_3c8f4_row63_col2" class="data row63 col2" >65536</td>
      <td id="T_3c8f4_row63_col3" class="data row63 col3" >65536</td>
      <td id="T_3c8f4_row63_col4" class="data row63 col4" >(128, 512, 1, 1)</td>
      <td id="T_3c8f4_row63_col5" class="data row63 col5" >layer2.3.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row64" class="row_heading level0 row64" >64</th>
      <td id="T_3c8f4_row64_col0" class="data row64 col0" >block_groups.1.block_group.3.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_3c8f4_row64_col1" class="data row64 col1" >(128,)</td>
      <td id="T_3c8f4_row64_col2" class="data row64 col2" >128</td>
      <td id="T_3c8f4_row64_col3" class="data row64 col3" >128</td>
      <td id="T_3c8f4_row64_col4" class="data row64 col4" >(128,)</td>
      <td id="T_3c8f4_row64_col5" class="data row64 col5" >layer2.3.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row65" class="row_heading level0 row65" >65</th>
      <td id="T_3c8f4_row65_col0" class="data row65 col0" >block_groups.1.block_group.3.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_3c8f4_row65_col1" class="data row65 col1" >(128,)</td>
      <td id="T_3c8f4_row65_col2" class="data row65 col2" >128</td>
      <td id="T_3c8f4_row65_col3" class="data row65 col3" >128</td>
      <td id="T_3c8f4_row65_col4" class="data row65 col4" >(128,)</td>
      <td id="T_3c8f4_row65_col5" class="data row65 col5" >layer2.3.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row66" class="row_heading level0 row66" >66</th>
      <td id="T_3c8f4_row66_col0" class="data row66 col0" >block_groups.1.block_group.3.left.conv_block.1.conv.weight</td>
      <td id="T_3c8f4_row66_col1" class="data row66 col1" >(128, 128, 3, 3)</td>
      <td id="T_3c8f4_row66_col2" class="data row66 col2" >147456</td>
      <td id="T_3c8f4_row66_col3" class="data row66 col3" >147456</td>
      <td id="T_3c8f4_row66_col4" class="data row66 col4" >(128, 128, 3, 3)</td>
      <td id="T_3c8f4_row66_col5" class="data row66 col5" >layer2.3.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row67" class="row_heading level0 row67" >67</th>
      <td id="T_3c8f4_row67_col0" class="data row67 col0" >block_groups.1.block_group.3.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_3c8f4_row67_col1" class="data row67 col1" >(128,)</td>
      <td id="T_3c8f4_row67_col2" class="data row67 col2" >128</td>
      <td id="T_3c8f4_row67_col3" class="data row67 col3" >128</td>
      <td id="T_3c8f4_row67_col4" class="data row67 col4" >(128,)</td>
      <td id="T_3c8f4_row67_col5" class="data row67 col5" >layer2.3.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row68" class="row_heading level0 row68" >68</th>
      <td id="T_3c8f4_row68_col0" class="data row68 col0" >block_groups.1.block_group.3.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_3c8f4_row68_col1" class="data row68 col1" >(128,)</td>
      <td id="T_3c8f4_row68_col2" class="data row68 col2" >128</td>
      <td id="T_3c8f4_row68_col3" class="data row68 col3" >128</td>
      <td id="T_3c8f4_row68_col4" class="data row68 col4" >(128,)</td>
      <td id="T_3c8f4_row68_col5" class="data row68 col5" >layer2.3.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row69" class="row_heading level0 row69" >69</th>
      <td id="T_3c8f4_row69_col0" class="data row69 col0" >block_groups.1.block_group.3.left.conv_block.2.conv.weight</td>
      <td id="T_3c8f4_row69_col1" class="data row69 col1" >(512, 128, 1, 1)</td>
      <td id="T_3c8f4_row69_col2" class="data row69 col2" >65536</td>
      <td id="T_3c8f4_row69_col3" class="data row69 col3" >65536</td>
      <td id="T_3c8f4_row69_col4" class="data row69 col4" >(512, 128, 1, 1)</td>
      <td id="T_3c8f4_row69_col5" class="data row69 col5" >layer2.3.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row70" class="row_heading level0 row70" >70</th>
      <td id="T_3c8f4_row70_col0" class="data row70 col0" >block_groups.1.block_group.3.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_3c8f4_row70_col1" class="data row70 col1" >(512,)</td>
      <td id="T_3c8f4_row70_col2" class="data row70 col2" >512</td>
      <td id="T_3c8f4_row70_col3" class="data row70 col3" >512</td>
      <td id="T_3c8f4_row70_col4" class="data row70 col4" >(512,)</td>
      <td id="T_3c8f4_row70_col5" class="data row70 col5" >layer2.3.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row71" class="row_heading level0 row71" >71</th>
      <td id="T_3c8f4_row71_col0" class="data row71 col0" >block_groups.1.block_group.3.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_3c8f4_row71_col1" class="data row71 col1" >(512,)</td>
      <td id="T_3c8f4_row71_col2" class="data row71 col2" >512</td>
      <td id="T_3c8f4_row71_col3" class="data row71 col3" >512</td>
      <td id="T_3c8f4_row71_col4" class="data row71 col4" >(512,)</td>
      <td id="T_3c8f4_row71_col5" class="data row71 col5" >layer2.3.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row72" class="row_heading level0 row72" >72</th>
      <td id="T_3c8f4_row72_col0" class="data row72 col0" >block_groups.2.block_group.0.left.conv_block.0.conv.weight</td>
      <td id="T_3c8f4_row72_col1" class="data row72 col1" >(256, 512, 1, 1)</td>
      <td id="T_3c8f4_row72_col2" class="data row72 col2" >131072</td>
      <td id="T_3c8f4_row72_col3" class="data row72 col3" >131072</td>
      <td id="T_3c8f4_row72_col4" class="data row72 col4" >(256, 512, 1, 1)</td>
      <td id="T_3c8f4_row72_col5" class="data row72 col5" >layer3.0.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row73" class="row_heading level0 row73" >73</th>
      <td id="T_3c8f4_row73_col0" class="data row73 col0" >block_groups.2.block_group.0.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_3c8f4_row73_col1" class="data row73 col1" >(256,)</td>
      <td id="T_3c8f4_row73_col2" class="data row73 col2" >256</td>
      <td id="T_3c8f4_row73_col3" class="data row73 col3" >256</td>
      <td id="T_3c8f4_row73_col4" class="data row73 col4" >(256,)</td>
      <td id="T_3c8f4_row73_col5" class="data row73 col5" >layer3.0.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row74" class="row_heading level0 row74" >74</th>
      <td id="T_3c8f4_row74_col0" class="data row74 col0" >block_groups.2.block_group.0.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_3c8f4_row74_col1" class="data row74 col1" >(256,)</td>
      <td id="T_3c8f4_row74_col2" class="data row74 col2" >256</td>
      <td id="T_3c8f4_row74_col3" class="data row74 col3" >256</td>
      <td id="T_3c8f4_row74_col4" class="data row74 col4" >(256,)</td>
      <td id="T_3c8f4_row74_col5" class="data row74 col5" >layer3.0.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row75" class="row_heading level0 row75" >75</th>
      <td id="T_3c8f4_row75_col0" class="data row75 col0" >block_groups.2.block_group.0.left.conv_block.1.conv.weight</td>
      <td id="T_3c8f4_row75_col1" class="data row75 col1" >(256, 256, 3, 3)</td>
      <td id="T_3c8f4_row75_col2" class="data row75 col2" >589824</td>
      <td id="T_3c8f4_row75_col3" class="data row75 col3" >589824</td>
      <td id="T_3c8f4_row75_col4" class="data row75 col4" >(256, 256, 3, 3)</td>
      <td id="T_3c8f4_row75_col5" class="data row75 col5" >layer3.0.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row76" class="row_heading level0 row76" >76</th>
      <td id="T_3c8f4_row76_col0" class="data row76 col0" >block_groups.2.block_group.0.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_3c8f4_row76_col1" class="data row76 col1" >(256,)</td>
      <td id="T_3c8f4_row76_col2" class="data row76 col2" >256</td>
      <td id="T_3c8f4_row76_col3" class="data row76 col3" >256</td>
      <td id="T_3c8f4_row76_col4" class="data row76 col4" >(256,)</td>
      <td id="T_3c8f4_row76_col5" class="data row76 col5" >layer3.0.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row77" class="row_heading level0 row77" >77</th>
      <td id="T_3c8f4_row77_col0" class="data row77 col0" >block_groups.2.block_group.0.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_3c8f4_row77_col1" class="data row77 col1" >(256,)</td>
      <td id="T_3c8f4_row77_col2" class="data row77 col2" >256</td>
      <td id="T_3c8f4_row77_col3" class="data row77 col3" >256</td>
      <td id="T_3c8f4_row77_col4" class="data row77 col4" >(256,)</td>
      <td id="T_3c8f4_row77_col5" class="data row77 col5" >layer3.0.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row78" class="row_heading level0 row78" >78</th>
      <td id="T_3c8f4_row78_col0" class="data row78 col0" >block_groups.2.block_group.0.left.conv_block.2.conv.weight</td>
      <td id="T_3c8f4_row78_col1" class="data row78 col1" >(1024, 256, 1, 1)</td>
      <td id="T_3c8f4_row78_col2" class="data row78 col2" >262144</td>
      <td id="T_3c8f4_row78_col3" class="data row78 col3" >262144</td>
      <td id="T_3c8f4_row78_col4" class="data row78 col4" >(1024, 256, 1, 1)</td>
      <td id="T_3c8f4_row78_col5" class="data row78 col5" >layer3.0.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row79" class="row_heading level0 row79" >79</th>
      <td id="T_3c8f4_row79_col0" class="data row79 col0" >block_groups.2.block_group.0.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_3c8f4_row79_col1" class="data row79 col1" >(1024,)</td>
      <td id="T_3c8f4_row79_col2" class="data row79 col2" >1024</td>
      <td id="T_3c8f4_row79_col3" class="data row79 col3" >1024</td>
      <td id="T_3c8f4_row79_col4" class="data row79 col4" >(1024,)</td>
      <td id="T_3c8f4_row79_col5" class="data row79 col5" >layer3.0.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row80" class="row_heading level0 row80" >80</th>
      <td id="T_3c8f4_row80_col0" class="data row80 col0" >block_groups.2.block_group.0.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_3c8f4_row80_col1" class="data row80 col1" >(1024,)</td>
      <td id="T_3c8f4_row80_col2" class="data row80 col2" >1024</td>
      <td id="T_3c8f4_row80_col3" class="data row80 col3" >1024</td>
      <td id="T_3c8f4_row80_col4" class="data row80 col4" >(1024,)</td>
      <td id="T_3c8f4_row80_col5" class="data row80 col5" >layer3.0.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row81" class="row_heading level0 row81" >81</th>
      <td id="T_3c8f4_row81_col0" class="data row81 col0" >block_groups.2.block_group.0.right.conv.weight</td>
      <td id="T_3c8f4_row81_col1" class="data row81 col1" >(1024, 512, 1, 1)</td>
      <td id="T_3c8f4_row81_col2" class="data row81 col2" >524288</td>
      <td id="T_3c8f4_row81_col3" class="data row81 col3" >524288</td>
      <td id="T_3c8f4_row81_col4" class="data row81 col4" >(1024, 512, 1, 1)</td>
      <td id="T_3c8f4_row81_col5" class="data row81 col5" >layer3.0.downsample.0.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row82" class="row_heading level0 row82" >82</th>
      <td id="T_3c8f4_row82_col0" class="data row82 col0" >block_groups.2.block_group.0.right.batchnorm2d.weight</td>
      <td id="T_3c8f4_row82_col1" class="data row82 col1" >(1024,)</td>
      <td id="T_3c8f4_row82_col2" class="data row82 col2" >1024</td>
      <td id="T_3c8f4_row82_col3" class="data row82 col3" >1024</td>
      <td id="T_3c8f4_row82_col4" class="data row82 col4" >(1024,)</td>
      <td id="T_3c8f4_row82_col5" class="data row82 col5" >layer3.0.downsample.1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row83" class="row_heading level0 row83" >83</th>
      <td id="T_3c8f4_row83_col0" class="data row83 col0" >block_groups.2.block_group.0.right.batchnorm2d.bias</td>
      <td id="T_3c8f4_row83_col1" class="data row83 col1" >(1024,)</td>
      <td id="T_3c8f4_row83_col2" class="data row83 col2" >1024</td>
      <td id="T_3c8f4_row83_col3" class="data row83 col3" >1024</td>
      <td id="T_3c8f4_row83_col4" class="data row83 col4" >(1024,)</td>
      <td id="T_3c8f4_row83_col5" class="data row83 col5" >layer3.0.downsample.1.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row84" class="row_heading level0 row84" >84</th>
      <td id="T_3c8f4_row84_col0" class="data row84 col0" >block_groups.2.block_group.1.left.conv_block.0.conv.weight</td>
      <td id="T_3c8f4_row84_col1" class="data row84 col1" >(256, 1024, 1, 1)</td>
      <td id="T_3c8f4_row84_col2" class="data row84 col2" >262144</td>
      <td id="T_3c8f4_row84_col3" class="data row84 col3" >262144</td>
      <td id="T_3c8f4_row84_col4" class="data row84 col4" >(256, 1024, 1, 1)</td>
      <td id="T_3c8f4_row84_col5" class="data row84 col5" >layer3.1.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row85" class="row_heading level0 row85" >85</th>
      <td id="T_3c8f4_row85_col0" class="data row85 col0" >block_groups.2.block_group.1.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_3c8f4_row85_col1" class="data row85 col1" >(256,)</td>
      <td id="T_3c8f4_row85_col2" class="data row85 col2" >256</td>
      <td id="T_3c8f4_row85_col3" class="data row85 col3" >256</td>
      <td id="T_3c8f4_row85_col4" class="data row85 col4" >(256,)</td>
      <td id="T_3c8f4_row85_col5" class="data row85 col5" >layer3.1.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row86" class="row_heading level0 row86" >86</th>
      <td id="T_3c8f4_row86_col0" class="data row86 col0" >block_groups.2.block_group.1.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_3c8f4_row86_col1" class="data row86 col1" >(256,)</td>
      <td id="T_3c8f4_row86_col2" class="data row86 col2" >256</td>
      <td id="T_3c8f4_row86_col3" class="data row86 col3" >256</td>
      <td id="T_3c8f4_row86_col4" class="data row86 col4" >(256,)</td>
      <td id="T_3c8f4_row86_col5" class="data row86 col5" >layer3.1.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row87" class="row_heading level0 row87" >87</th>
      <td id="T_3c8f4_row87_col0" class="data row87 col0" >block_groups.2.block_group.1.left.conv_block.1.conv.weight</td>
      <td id="T_3c8f4_row87_col1" class="data row87 col1" >(256, 256, 3, 3)</td>
      <td id="T_3c8f4_row87_col2" class="data row87 col2" >589824</td>
      <td id="T_3c8f4_row87_col3" class="data row87 col3" >589824</td>
      <td id="T_3c8f4_row87_col4" class="data row87 col4" >(256, 256, 3, 3)</td>
      <td id="T_3c8f4_row87_col5" class="data row87 col5" >layer3.1.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row88" class="row_heading level0 row88" >88</th>
      <td id="T_3c8f4_row88_col0" class="data row88 col0" >block_groups.2.block_group.1.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_3c8f4_row88_col1" class="data row88 col1" >(256,)</td>
      <td id="T_3c8f4_row88_col2" class="data row88 col2" >256</td>
      <td id="T_3c8f4_row88_col3" class="data row88 col3" >256</td>
      <td id="T_3c8f4_row88_col4" class="data row88 col4" >(256,)</td>
      <td id="T_3c8f4_row88_col5" class="data row88 col5" >layer3.1.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row89" class="row_heading level0 row89" >89</th>
      <td id="T_3c8f4_row89_col0" class="data row89 col0" >block_groups.2.block_group.1.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_3c8f4_row89_col1" class="data row89 col1" >(256,)</td>
      <td id="T_3c8f4_row89_col2" class="data row89 col2" >256</td>
      <td id="T_3c8f4_row89_col3" class="data row89 col3" >256</td>
      <td id="T_3c8f4_row89_col4" class="data row89 col4" >(256,)</td>
      <td id="T_3c8f4_row89_col5" class="data row89 col5" >layer3.1.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row90" class="row_heading level0 row90" >90</th>
      <td id="T_3c8f4_row90_col0" class="data row90 col0" >block_groups.2.block_group.1.left.conv_block.2.conv.weight</td>
      <td id="T_3c8f4_row90_col1" class="data row90 col1" >(1024, 256, 1, 1)</td>
      <td id="T_3c8f4_row90_col2" class="data row90 col2" >262144</td>
      <td id="T_3c8f4_row90_col3" class="data row90 col3" >262144</td>
      <td id="T_3c8f4_row90_col4" class="data row90 col4" >(1024, 256, 1, 1)</td>
      <td id="T_3c8f4_row90_col5" class="data row90 col5" >layer3.1.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row91" class="row_heading level0 row91" >91</th>
      <td id="T_3c8f4_row91_col0" class="data row91 col0" >block_groups.2.block_group.1.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_3c8f4_row91_col1" class="data row91 col1" >(1024,)</td>
      <td id="T_3c8f4_row91_col2" class="data row91 col2" >1024</td>
      <td id="T_3c8f4_row91_col3" class="data row91 col3" >1024</td>
      <td id="T_3c8f4_row91_col4" class="data row91 col4" >(1024,)</td>
      <td id="T_3c8f4_row91_col5" class="data row91 col5" >layer3.1.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row92" class="row_heading level0 row92" >92</th>
      <td id="T_3c8f4_row92_col0" class="data row92 col0" >block_groups.2.block_group.1.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_3c8f4_row92_col1" class="data row92 col1" >(1024,)</td>
      <td id="T_3c8f4_row92_col2" class="data row92 col2" >1024</td>
      <td id="T_3c8f4_row92_col3" class="data row92 col3" >1024</td>
      <td id="T_3c8f4_row92_col4" class="data row92 col4" >(1024,)</td>
      <td id="T_3c8f4_row92_col5" class="data row92 col5" >layer3.1.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row93" class="row_heading level0 row93" >93</th>
      <td id="T_3c8f4_row93_col0" class="data row93 col0" >block_groups.2.block_group.2.left.conv_block.0.conv.weight</td>
      <td id="T_3c8f4_row93_col1" class="data row93 col1" >(256, 1024, 1, 1)</td>
      <td id="T_3c8f4_row93_col2" class="data row93 col2" >262144</td>
      <td id="T_3c8f4_row93_col3" class="data row93 col3" >262144</td>
      <td id="T_3c8f4_row93_col4" class="data row93 col4" >(256, 1024, 1, 1)</td>
      <td id="T_3c8f4_row93_col5" class="data row93 col5" >layer3.2.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row94" class="row_heading level0 row94" >94</th>
      <td id="T_3c8f4_row94_col0" class="data row94 col0" >block_groups.2.block_group.2.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_3c8f4_row94_col1" class="data row94 col1" >(256,)</td>
      <td id="T_3c8f4_row94_col2" class="data row94 col2" >256</td>
      <td id="T_3c8f4_row94_col3" class="data row94 col3" >256</td>
      <td id="T_3c8f4_row94_col4" class="data row94 col4" >(256,)</td>
      <td id="T_3c8f4_row94_col5" class="data row94 col5" >layer3.2.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row95" class="row_heading level0 row95" >95</th>
      <td id="T_3c8f4_row95_col0" class="data row95 col0" >block_groups.2.block_group.2.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_3c8f4_row95_col1" class="data row95 col1" >(256,)</td>
      <td id="T_3c8f4_row95_col2" class="data row95 col2" >256</td>
      <td id="T_3c8f4_row95_col3" class="data row95 col3" >256</td>
      <td id="T_3c8f4_row95_col4" class="data row95 col4" >(256,)</td>
      <td id="T_3c8f4_row95_col5" class="data row95 col5" >layer3.2.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row96" class="row_heading level0 row96" >96</th>
      <td id="T_3c8f4_row96_col0" class="data row96 col0" >block_groups.2.block_group.2.left.conv_block.1.conv.weight</td>
      <td id="T_3c8f4_row96_col1" class="data row96 col1" >(256, 256, 3, 3)</td>
      <td id="T_3c8f4_row96_col2" class="data row96 col2" >589824</td>
      <td id="T_3c8f4_row96_col3" class="data row96 col3" >589824</td>
      <td id="T_3c8f4_row96_col4" class="data row96 col4" >(256, 256, 3, 3)</td>
      <td id="T_3c8f4_row96_col5" class="data row96 col5" >layer3.2.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row97" class="row_heading level0 row97" >97</th>
      <td id="T_3c8f4_row97_col0" class="data row97 col0" >block_groups.2.block_group.2.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_3c8f4_row97_col1" class="data row97 col1" >(256,)</td>
      <td id="T_3c8f4_row97_col2" class="data row97 col2" >256</td>
      <td id="T_3c8f4_row97_col3" class="data row97 col3" >256</td>
      <td id="T_3c8f4_row97_col4" class="data row97 col4" >(256,)</td>
      <td id="T_3c8f4_row97_col5" class="data row97 col5" >layer3.2.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row98" class="row_heading level0 row98" >98</th>
      <td id="T_3c8f4_row98_col0" class="data row98 col0" >block_groups.2.block_group.2.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_3c8f4_row98_col1" class="data row98 col1" >(256,)</td>
      <td id="T_3c8f4_row98_col2" class="data row98 col2" >256</td>
      <td id="T_3c8f4_row98_col3" class="data row98 col3" >256</td>
      <td id="T_3c8f4_row98_col4" class="data row98 col4" >(256,)</td>
      <td id="T_3c8f4_row98_col5" class="data row98 col5" >layer3.2.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row99" class="row_heading level0 row99" >99</th>
      <td id="T_3c8f4_row99_col0" class="data row99 col0" >block_groups.2.block_group.2.left.conv_block.2.conv.weight</td>
      <td id="T_3c8f4_row99_col1" class="data row99 col1" >(1024, 256, 1, 1)</td>
      <td id="T_3c8f4_row99_col2" class="data row99 col2" >262144</td>
      <td id="T_3c8f4_row99_col3" class="data row99 col3" >262144</td>
      <td id="T_3c8f4_row99_col4" class="data row99 col4" >(1024, 256, 1, 1)</td>
      <td id="T_3c8f4_row99_col5" class="data row99 col5" >layer3.2.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row100" class="row_heading level0 row100" >100</th>
      <td id="T_3c8f4_row100_col0" class="data row100 col0" >block_groups.2.block_group.2.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_3c8f4_row100_col1" class="data row100 col1" >(1024,)</td>
      <td id="T_3c8f4_row100_col2" class="data row100 col2" >1024</td>
      <td id="T_3c8f4_row100_col3" class="data row100 col3" >1024</td>
      <td id="T_3c8f4_row100_col4" class="data row100 col4" >(1024,)</td>
      <td id="T_3c8f4_row100_col5" class="data row100 col5" >layer3.2.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row101" class="row_heading level0 row101" >101</th>
      <td id="T_3c8f4_row101_col0" class="data row101 col0" >block_groups.2.block_group.2.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_3c8f4_row101_col1" class="data row101 col1" >(1024,)</td>
      <td id="T_3c8f4_row101_col2" class="data row101 col2" >1024</td>
      <td id="T_3c8f4_row101_col3" class="data row101 col3" >1024</td>
      <td id="T_3c8f4_row101_col4" class="data row101 col4" >(1024,)</td>
      <td id="T_3c8f4_row101_col5" class="data row101 col5" >layer3.2.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row102" class="row_heading level0 row102" >102</th>
      <td id="T_3c8f4_row102_col0" class="data row102 col0" >block_groups.2.block_group.3.left.conv_block.0.conv.weight</td>
      <td id="T_3c8f4_row102_col1" class="data row102 col1" >(256, 1024, 1, 1)</td>
      <td id="T_3c8f4_row102_col2" class="data row102 col2" >262144</td>
      <td id="T_3c8f4_row102_col3" class="data row102 col3" >262144</td>
      <td id="T_3c8f4_row102_col4" class="data row102 col4" >(256, 1024, 1, 1)</td>
      <td id="T_3c8f4_row102_col5" class="data row102 col5" >layer3.3.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row103" class="row_heading level0 row103" >103</th>
      <td id="T_3c8f4_row103_col0" class="data row103 col0" >block_groups.2.block_group.3.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_3c8f4_row103_col1" class="data row103 col1" >(256,)</td>
      <td id="T_3c8f4_row103_col2" class="data row103 col2" >256</td>
      <td id="T_3c8f4_row103_col3" class="data row103 col3" >256</td>
      <td id="T_3c8f4_row103_col4" class="data row103 col4" >(256,)</td>
      <td id="T_3c8f4_row103_col5" class="data row103 col5" >layer3.3.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row104" class="row_heading level0 row104" >104</th>
      <td id="T_3c8f4_row104_col0" class="data row104 col0" >block_groups.2.block_group.3.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_3c8f4_row104_col1" class="data row104 col1" >(256,)</td>
      <td id="T_3c8f4_row104_col2" class="data row104 col2" >256</td>
      <td id="T_3c8f4_row104_col3" class="data row104 col3" >256</td>
      <td id="T_3c8f4_row104_col4" class="data row104 col4" >(256,)</td>
      <td id="T_3c8f4_row104_col5" class="data row104 col5" >layer3.3.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row105" class="row_heading level0 row105" >105</th>
      <td id="T_3c8f4_row105_col0" class="data row105 col0" >block_groups.2.block_group.3.left.conv_block.1.conv.weight</td>
      <td id="T_3c8f4_row105_col1" class="data row105 col1" >(256, 256, 3, 3)</td>
      <td id="T_3c8f4_row105_col2" class="data row105 col2" >589824</td>
      <td id="T_3c8f4_row105_col3" class="data row105 col3" >589824</td>
      <td id="T_3c8f4_row105_col4" class="data row105 col4" >(256, 256, 3, 3)</td>
      <td id="T_3c8f4_row105_col5" class="data row105 col5" >layer3.3.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row106" class="row_heading level0 row106" >106</th>
      <td id="T_3c8f4_row106_col0" class="data row106 col0" >block_groups.2.block_group.3.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_3c8f4_row106_col1" class="data row106 col1" >(256,)</td>
      <td id="T_3c8f4_row106_col2" class="data row106 col2" >256</td>
      <td id="T_3c8f4_row106_col3" class="data row106 col3" >256</td>
      <td id="T_3c8f4_row106_col4" class="data row106 col4" >(256,)</td>
      <td id="T_3c8f4_row106_col5" class="data row106 col5" >layer3.3.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row107" class="row_heading level0 row107" >107</th>
      <td id="T_3c8f4_row107_col0" class="data row107 col0" >block_groups.2.block_group.3.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_3c8f4_row107_col1" class="data row107 col1" >(256,)</td>
      <td id="T_3c8f4_row107_col2" class="data row107 col2" >256</td>
      <td id="T_3c8f4_row107_col3" class="data row107 col3" >256</td>
      <td id="T_3c8f4_row107_col4" class="data row107 col4" >(256,)</td>
      <td id="T_3c8f4_row107_col5" class="data row107 col5" >layer3.3.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row108" class="row_heading level0 row108" >108</th>
      <td id="T_3c8f4_row108_col0" class="data row108 col0" >block_groups.2.block_group.3.left.conv_block.2.conv.weight</td>
      <td id="T_3c8f4_row108_col1" class="data row108 col1" >(1024, 256, 1, 1)</td>
      <td id="T_3c8f4_row108_col2" class="data row108 col2" >262144</td>
      <td id="T_3c8f4_row108_col3" class="data row108 col3" >262144</td>
      <td id="T_3c8f4_row108_col4" class="data row108 col4" >(1024, 256, 1, 1)</td>
      <td id="T_3c8f4_row108_col5" class="data row108 col5" >layer3.3.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row109" class="row_heading level0 row109" >109</th>
      <td id="T_3c8f4_row109_col0" class="data row109 col0" >block_groups.2.block_group.3.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_3c8f4_row109_col1" class="data row109 col1" >(1024,)</td>
      <td id="T_3c8f4_row109_col2" class="data row109 col2" >1024</td>
      <td id="T_3c8f4_row109_col3" class="data row109 col3" >1024</td>
      <td id="T_3c8f4_row109_col4" class="data row109 col4" >(1024,)</td>
      <td id="T_3c8f4_row109_col5" class="data row109 col5" >layer3.3.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row110" class="row_heading level0 row110" >110</th>
      <td id="T_3c8f4_row110_col0" class="data row110 col0" >block_groups.2.block_group.3.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_3c8f4_row110_col1" class="data row110 col1" >(1024,)</td>
      <td id="T_3c8f4_row110_col2" class="data row110 col2" >1024</td>
      <td id="T_3c8f4_row110_col3" class="data row110 col3" >1024</td>
      <td id="T_3c8f4_row110_col4" class="data row110 col4" >(1024,)</td>
      <td id="T_3c8f4_row110_col5" class="data row110 col5" >layer3.3.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row111" class="row_heading level0 row111" >111</th>
      <td id="T_3c8f4_row111_col0" class="data row111 col0" >block_groups.2.block_group.4.left.conv_block.0.conv.weight</td>
      <td id="T_3c8f4_row111_col1" class="data row111 col1" >(256, 1024, 1, 1)</td>
      <td id="T_3c8f4_row111_col2" class="data row111 col2" >262144</td>
      <td id="T_3c8f4_row111_col3" class="data row111 col3" >262144</td>
      <td id="T_3c8f4_row111_col4" class="data row111 col4" >(256, 1024, 1, 1)</td>
      <td id="T_3c8f4_row111_col5" class="data row111 col5" >layer3.4.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row112" class="row_heading level0 row112" >112</th>
      <td id="T_3c8f4_row112_col0" class="data row112 col0" >block_groups.2.block_group.4.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_3c8f4_row112_col1" class="data row112 col1" >(256,)</td>
      <td id="T_3c8f4_row112_col2" class="data row112 col2" >256</td>
      <td id="T_3c8f4_row112_col3" class="data row112 col3" >256</td>
      <td id="T_3c8f4_row112_col4" class="data row112 col4" >(256,)</td>
      <td id="T_3c8f4_row112_col5" class="data row112 col5" >layer3.4.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row113" class="row_heading level0 row113" >113</th>
      <td id="T_3c8f4_row113_col0" class="data row113 col0" >block_groups.2.block_group.4.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_3c8f4_row113_col1" class="data row113 col1" >(256,)</td>
      <td id="T_3c8f4_row113_col2" class="data row113 col2" >256</td>
      <td id="T_3c8f4_row113_col3" class="data row113 col3" >256</td>
      <td id="T_3c8f4_row113_col4" class="data row113 col4" >(256,)</td>
      <td id="T_3c8f4_row113_col5" class="data row113 col5" >layer3.4.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row114" class="row_heading level0 row114" >114</th>
      <td id="T_3c8f4_row114_col0" class="data row114 col0" >block_groups.2.block_group.4.left.conv_block.1.conv.weight</td>
      <td id="T_3c8f4_row114_col1" class="data row114 col1" >(256, 256, 3, 3)</td>
      <td id="T_3c8f4_row114_col2" class="data row114 col2" >589824</td>
      <td id="T_3c8f4_row114_col3" class="data row114 col3" >589824</td>
      <td id="T_3c8f4_row114_col4" class="data row114 col4" >(256, 256, 3, 3)</td>
      <td id="T_3c8f4_row114_col5" class="data row114 col5" >layer3.4.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row115" class="row_heading level0 row115" >115</th>
      <td id="T_3c8f4_row115_col0" class="data row115 col0" >block_groups.2.block_group.4.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_3c8f4_row115_col1" class="data row115 col1" >(256,)</td>
      <td id="T_3c8f4_row115_col2" class="data row115 col2" >256</td>
      <td id="T_3c8f4_row115_col3" class="data row115 col3" >256</td>
      <td id="T_3c8f4_row115_col4" class="data row115 col4" >(256,)</td>
      <td id="T_3c8f4_row115_col5" class="data row115 col5" >layer3.4.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row116" class="row_heading level0 row116" >116</th>
      <td id="T_3c8f4_row116_col0" class="data row116 col0" >block_groups.2.block_group.4.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_3c8f4_row116_col1" class="data row116 col1" >(256,)</td>
      <td id="T_3c8f4_row116_col2" class="data row116 col2" >256</td>
      <td id="T_3c8f4_row116_col3" class="data row116 col3" >256</td>
      <td id="T_3c8f4_row116_col4" class="data row116 col4" >(256,)</td>
      <td id="T_3c8f4_row116_col5" class="data row116 col5" >layer3.4.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row117" class="row_heading level0 row117" >117</th>
      <td id="T_3c8f4_row117_col0" class="data row117 col0" >block_groups.2.block_group.4.left.conv_block.2.conv.weight</td>
      <td id="T_3c8f4_row117_col1" class="data row117 col1" >(1024, 256, 1, 1)</td>
      <td id="T_3c8f4_row117_col2" class="data row117 col2" >262144</td>
      <td id="T_3c8f4_row117_col3" class="data row117 col3" >262144</td>
      <td id="T_3c8f4_row117_col4" class="data row117 col4" >(1024, 256, 1, 1)</td>
      <td id="T_3c8f4_row117_col5" class="data row117 col5" >layer3.4.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row118" class="row_heading level0 row118" >118</th>
      <td id="T_3c8f4_row118_col0" class="data row118 col0" >block_groups.2.block_group.4.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_3c8f4_row118_col1" class="data row118 col1" >(1024,)</td>
      <td id="T_3c8f4_row118_col2" class="data row118 col2" >1024</td>
      <td id="T_3c8f4_row118_col3" class="data row118 col3" >1024</td>
      <td id="T_3c8f4_row118_col4" class="data row118 col4" >(1024,)</td>
      <td id="T_3c8f4_row118_col5" class="data row118 col5" >layer3.4.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row119" class="row_heading level0 row119" >119</th>
      <td id="T_3c8f4_row119_col0" class="data row119 col0" >block_groups.2.block_group.4.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_3c8f4_row119_col1" class="data row119 col1" >(1024,)</td>
      <td id="T_3c8f4_row119_col2" class="data row119 col2" >1024</td>
      <td id="T_3c8f4_row119_col3" class="data row119 col3" >1024</td>
      <td id="T_3c8f4_row119_col4" class="data row119 col4" >(1024,)</td>
      <td id="T_3c8f4_row119_col5" class="data row119 col5" >layer3.4.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row120" class="row_heading level0 row120" >120</th>
      <td id="T_3c8f4_row120_col0" class="data row120 col0" >block_groups.2.block_group.5.left.conv_block.0.conv.weight</td>
      <td id="T_3c8f4_row120_col1" class="data row120 col1" >(256, 1024, 1, 1)</td>
      <td id="T_3c8f4_row120_col2" class="data row120 col2" >262144</td>
      <td id="T_3c8f4_row120_col3" class="data row120 col3" >262144</td>
      <td id="T_3c8f4_row120_col4" class="data row120 col4" >(256, 1024, 1, 1)</td>
      <td id="T_3c8f4_row120_col5" class="data row120 col5" >layer3.5.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row121" class="row_heading level0 row121" >121</th>
      <td id="T_3c8f4_row121_col0" class="data row121 col0" >block_groups.2.block_group.5.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_3c8f4_row121_col1" class="data row121 col1" >(256,)</td>
      <td id="T_3c8f4_row121_col2" class="data row121 col2" >256</td>
      <td id="T_3c8f4_row121_col3" class="data row121 col3" >256</td>
      <td id="T_3c8f4_row121_col4" class="data row121 col4" >(256,)</td>
      <td id="T_3c8f4_row121_col5" class="data row121 col5" >layer3.5.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row122" class="row_heading level0 row122" >122</th>
      <td id="T_3c8f4_row122_col0" class="data row122 col0" >block_groups.2.block_group.5.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_3c8f4_row122_col1" class="data row122 col1" >(256,)</td>
      <td id="T_3c8f4_row122_col2" class="data row122 col2" >256</td>
      <td id="T_3c8f4_row122_col3" class="data row122 col3" >256</td>
      <td id="T_3c8f4_row122_col4" class="data row122 col4" >(256,)</td>
      <td id="T_3c8f4_row122_col5" class="data row122 col5" >layer3.5.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row123" class="row_heading level0 row123" >123</th>
      <td id="T_3c8f4_row123_col0" class="data row123 col0" >block_groups.2.block_group.5.left.conv_block.1.conv.weight</td>
      <td id="T_3c8f4_row123_col1" class="data row123 col1" >(256, 256, 3, 3)</td>
      <td id="T_3c8f4_row123_col2" class="data row123 col2" >589824</td>
      <td id="T_3c8f4_row123_col3" class="data row123 col3" >589824</td>
      <td id="T_3c8f4_row123_col4" class="data row123 col4" >(256, 256, 3, 3)</td>
      <td id="T_3c8f4_row123_col5" class="data row123 col5" >layer3.5.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row124" class="row_heading level0 row124" >124</th>
      <td id="T_3c8f4_row124_col0" class="data row124 col0" >block_groups.2.block_group.5.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_3c8f4_row124_col1" class="data row124 col1" >(256,)</td>
      <td id="T_3c8f4_row124_col2" class="data row124 col2" >256</td>
      <td id="T_3c8f4_row124_col3" class="data row124 col3" >256</td>
      <td id="T_3c8f4_row124_col4" class="data row124 col4" >(256,)</td>
      <td id="T_3c8f4_row124_col5" class="data row124 col5" >layer3.5.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row125" class="row_heading level0 row125" >125</th>
      <td id="T_3c8f4_row125_col0" class="data row125 col0" >block_groups.2.block_group.5.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_3c8f4_row125_col1" class="data row125 col1" >(256,)</td>
      <td id="T_3c8f4_row125_col2" class="data row125 col2" >256</td>
      <td id="T_3c8f4_row125_col3" class="data row125 col3" >256</td>
      <td id="T_3c8f4_row125_col4" class="data row125 col4" >(256,)</td>
      <td id="T_3c8f4_row125_col5" class="data row125 col5" >layer3.5.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row126" class="row_heading level0 row126" >126</th>
      <td id="T_3c8f4_row126_col0" class="data row126 col0" >block_groups.2.block_group.5.left.conv_block.2.conv.weight</td>
      <td id="T_3c8f4_row126_col1" class="data row126 col1" >(1024, 256, 1, 1)</td>
      <td id="T_3c8f4_row126_col2" class="data row126 col2" >262144</td>
      <td id="T_3c8f4_row126_col3" class="data row126 col3" >262144</td>
      <td id="T_3c8f4_row126_col4" class="data row126 col4" >(1024, 256, 1, 1)</td>
      <td id="T_3c8f4_row126_col5" class="data row126 col5" >layer3.5.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row127" class="row_heading level0 row127" >127</th>
      <td id="T_3c8f4_row127_col0" class="data row127 col0" >block_groups.2.block_group.5.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_3c8f4_row127_col1" class="data row127 col1" >(1024,)</td>
      <td id="T_3c8f4_row127_col2" class="data row127 col2" >1024</td>
      <td id="T_3c8f4_row127_col3" class="data row127 col3" >1024</td>
      <td id="T_3c8f4_row127_col4" class="data row127 col4" >(1024,)</td>
      <td id="T_3c8f4_row127_col5" class="data row127 col5" >layer3.5.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row128" class="row_heading level0 row128" >128</th>
      <td id="T_3c8f4_row128_col0" class="data row128 col0" >block_groups.2.block_group.5.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_3c8f4_row128_col1" class="data row128 col1" >(1024,)</td>
      <td id="T_3c8f4_row128_col2" class="data row128 col2" >1024</td>
      <td id="T_3c8f4_row128_col3" class="data row128 col3" >1024</td>
      <td id="T_3c8f4_row128_col4" class="data row128 col4" >(1024,)</td>
      <td id="T_3c8f4_row128_col5" class="data row128 col5" >layer3.5.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row129" class="row_heading level0 row129" >129</th>
      <td id="T_3c8f4_row129_col0" class="data row129 col0" >block_groups.3.block_group.0.left.conv_block.0.conv.weight</td>
      <td id="T_3c8f4_row129_col1" class="data row129 col1" >(512, 1024, 1, 1)</td>
      <td id="T_3c8f4_row129_col2" class="data row129 col2" >524288</td>
      <td id="T_3c8f4_row129_col3" class="data row129 col3" >524288</td>
      <td id="T_3c8f4_row129_col4" class="data row129 col4" >(512, 1024, 1, 1)</td>
      <td id="T_3c8f4_row129_col5" class="data row129 col5" >layer4.0.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row130" class="row_heading level0 row130" >130</th>
      <td id="T_3c8f4_row130_col0" class="data row130 col0" >block_groups.3.block_group.0.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_3c8f4_row130_col1" class="data row130 col1" >(512,)</td>
      <td id="T_3c8f4_row130_col2" class="data row130 col2" >512</td>
      <td id="T_3c8f4_row130_col3" class="data row130 col3" >512</td>
      <td id="T_3c8f4_row130_col4" class="data row130 col4" >(512,)</td>
      <td id="T_3c8f4_row130_col5" class="data row130 col5" >layer4.0.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row131" class="row_heading level0 row131" >131</th>
      <td id="T_3c8f4_row131_col0" class="data row131 col0" >block_groups.3.block_group.0.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_3c8f4_row131_col1" class="data row131 col1" >(512,)</td>
      <td id="T_3c8f4_row131_col2" class="data row131 col2" >512</td>
      <td id="T_3c8f4_row131_col3" class="data row131 col3" >512</td>
      <td id="T_3c8f4_row131_col4" class="data row131 col4" >(512,)</td>
      <td id="T_3c8f4_row131_col5" class="data row131 col5" >layer4.0.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row132" class="row_heading level0 row132" >132</th>
      <td id="T_3c8f4_row132_col0" class="data row132 col0" >block_groups.3.block_group.0.left.conv_block.1.conv.weight</td>
      <td id="T_3c8f4_row132_col1" class="data row132 col1" >(512, 512, 3, 3)</td>
      <td id="T_3c8f4_row132_col2" class="data row132 col2" >2359296</td>
      <td id="T_3c8f4_row132_col3" class="data row132 col3" >2359296</td>
      <td id="T_3c8f4_row132_col4" class="data row132 col4" >(512, 512, 3, 3)</td>
      <td id="T_3c8f4_row132_col5" class="data row132 col5" >layer4.0.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row133" class="row_heading level0 row133" >133</th>
      <td id="T_3c8f4_row133_col0" class="data row133 col0" >block_groups.3.block_group.0.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_3c8f4_row133_col1" class="data row133 col1" >(512,)</td>
      <td id="T_3c8f4_row133_col2" class="data row133 col2" >512</td>
      <td id="T_3c8f4_row133_col3" class="data row133 col3" >512</td>
      <td id="T_3c8f4_row133_col4" class="data row133 col4" >(512,)</td>
      <td id="T_3c8f4_row133_col5" class="data row133 col5" >layer4.0.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row134" class="row_heading level0 row134" >134</th>
      <td id="T_3c8f4_row134_col0" class="data row134 col0" >block_groups.3.block_group.0.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_3c8f4_row134_col1" class="data row134 col1" >(512,)</td>
      <td id="T_3c8f4_row134_col2" class="data row134 col2" >512</td>
      <td id="T_3c8f4_row134_col3" class="data row134 col3" >512</td>
      <td id="T_3c8f4_row134_col4" class="data row134 col4" >(512,)</td>
      <td id="T_3c8f4_row134_col5" class="data row134 col5" >layer4.0.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row135" class="row_heading level0 row135" >135</th>
      <td id="T_3c8f4_row135_col0" class="data row135 col0" >block_groups.3.block_group.0.left.conv_block.2.conv.weight</td>
      <td id="T_3c8f4_row135_col1" class="data row135 col1" >(2048, 512, 1, 1)</td>
      <td id="T_3c8f4_row135_col2" class="data row135 col2" >1048576</td>
      <td id="T_3c8f4_row135_col3" class="data row135 col3" >1048576</td>
      <td id="T_3c8f4_row135_col4" class="data row135 col4" >(2048, 512, 1, 1)</td>
      <td id="T_3c8f4_row135_col5" class="data row135 col5" >layer4.0.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row136" class="row_heading level0 row136" >136</th>
      <td id="T_3c8f4_row136_col0" class="data row136 col0" >block_groups.3.block_group.0.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_3c8f4_row136_col1" class="data row136 col1" >(2048,)</td>
      <td id="T_3c8f4_row136_col2" class="data row136 col2" >2048</td>
      <td id="T_3c8f4_row136_col3" class="data row136 col3" >2048</td>
      <td id="T_3c8f4_row136_col4" class="data row136 col4" >(2048,)</td>
      <td id="T_3c8f4_row136_col5" class="data row136 col5" >layer4.0.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row137" class="row_heading level0 row137" >137</th>
      <td id="T_3c8f4_row137_col0" class="data row137 col0" >block_groups.3.block_group.0.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_3c8f4_row137_col1" class="data row137 col1" >(2048,)</td>
      <td id="T_3c8f4_row137_col2" class="data row137 col2" >2048</td>
      <td id="T_3c8f4_row137_col3" class="data row137 col3" >2048</td>
      <td id="T_3c8f4_row137_col4" class="data row137 col4" >(2048,)</td>
      <td id="T_3c8f4_row137_col5" class="data row137 col5" >layer4.0.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row138" class="row_heading level0 row138" >138</th>
      <td id="T_3c8f4_row138_col0" class="data row138 col0" >block_groups.3.block_group.0.right.conv.weight</td>
      <td id="T_3c8f4_row138_col1" class="data row138 col1" >(2048, 1024, 1, 1)</td>
      <td id="T_3c8f4_row138_col2" class="data row138 col2" >2097152</td>
      <td id="T_3c8f4_row138_col3" class="data row138 col3" >2097152</td>
      <td id="T_3c8f4_row138_col4" class="data row138 col4" >(2048, 1024, 1, 1)</td>
      <td id="T_3c8f4_row138_col5" class="data row138 col5" >layer4.0.downsample.0.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row139" class="row_heading level0 row139" >139</th>
      <td id="T_3c8f4_row139_col0" class="data row139 col0" >block_groups.3.block_group.0.right.batchnorm2d.weight</td>
      <td id="T_3c8f4_row139_col1" class="data row139 col1" >(2048,)</td>
      <td id="T_3c8f4_row139_col2" class="data row139 col2" >2048</td>
      <td id="T_3c8f4_row139_col3" class="data row139 col3" >2048</td>
      <td id="T_3c8f4_row139_col4" class="data row139 col4" >(2048,)</td>
      <td id="T_3c8f4_row139_col5" class="data row139 col5" >layer4.0.downsample.1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row140" class="row_heading level0 row140" >140</th>
      <td id="T_3c8f4_row140_col0" class="data row140 col0" >block_groups.3.block_group.0.right.batchnorm2d.bias</td>
      <td id="T_3c8f4_row140_col1" class="data row140 col1" >(2048,)</td>
      <td id="T_3c8f4_row140_col2" class="data row140 col2" >2048</td>
      <td id="T_3c8f4_row140_col3" class="data row140 col3" >2048</td>
      <td id="T_3c8f4_row140_col4" class="data row140 col4" >(2048,)</td>
      <td id="T_3c8f4_row140_col5" class="data row140 col5" >layer4.0.downsample.1.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row141" class="row_heading level0 row141" >141</th>
      <td id="T_3c8f4_row141_col0" class="data row141 col0" >block_groups.3.block_group.1.left.conv_block.0.conv.weight</td>
      <td id="T_3c8f4_row141_col1" class="data row141 col1" >(512, 2048, 1, 1)</td>
      <td id="T_3c8f4_row141_col2" class="data row141 col2" >1048576</td>
      <td id="T_3c8f4_row141_col3" class="data row141 col3" >1048576</td>
      <td id="T_3c8f4_row141_col4" class="data row141 col4" >(512, 2048, 1, 1)</td>
      <td id="T_3c8f4_row141_col5" class="data row141 col5" >layer4.1.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row142" class="row_heading level0 row142" >142</th>
      <td id="T_3c8f4_row142_col0" class="data row142 col0" >block_groups.3.block_group.1.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_3c8f4_row142_col1" class="data row142 col1" >(512,)</td>
      <td id="T_3c8f4_row142_col2" class="data row142 col2" >512</td>
      <td id="T_3c8f4_row142_col3" class="data row142 col3" >512</td>
      <td id="T_3c8f4_row142_col4" class="data row142 col4" >(512,)</td>
      <td id="T_3c8f4_row142_col5" class="data row142 col5" >layer4.1.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row143" class="row_heading level0 row143" >143</th>
      <td id="T_3c8f4_row143_col0" class="data row143 col0" >block_groups.3.block_group.1.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_3c8f4_row143_col1" class="data row143 col1" >(512,)</td>
      <td id="T_3c8f4_row143_col2" class="data row143 col2" >512</td>
      <td id="T_3c8f4_row143_col3" class="data row143 col3" >512</td>
      <td id="T_3c8f4_row143_col4" class="data row143 col4" >(512,)</td>
      <td id="T_3c8f4_row143_col5" class="data row143 col5" >layer4.1.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row144" class="row_heading level0 row144" >144</th>
      <td id="T_3c8f4_row144_col0" class="data row144 col0" >block_groups.3.block_group.1.left.conv_block.1.conv.weight</td>
      <td id="T_3c8f4_row144_col1" class="data row144 col1" >(512, 512, 3, 3)</td>
      <td id="T_3c8f4_row144_col2" class="data row144 col2" >2359296</td>
      <td id="T_3c8f4_row144_col3" class="data row144 col3" >2359296</td>
      <td id="T_3c8f4_row144_col4" class="data row144 col4" >(512, 512, 3, 3)</td>
      <td id="T_3c8f4_row144_col5" class="data row144 col5" >layer4.1.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row145" class="row_heading level0 row145" >145</th>
      <td id="T_3c8f4_row145_col0" class="data row145 col0" >block_groups.3.block_group.1.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_3c8f4_row145_col1" class="data row145 col1" >(512,)</td>
      <td id="T_3c8f4_row145_col2" class="data row145 col2" >512</td>
      <td id="T_3c8f4_row145_col3" class="data row145 col3" >512</td>
      <td id="T_3c8f4_row145_col4" class="data row145 col4" >(512,)</td>
      <td id="T_3c8f4_row145_col5" class="data row145 col5" >layer4.1.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row146" class="row_heading level0 row146" >146</th>
      <td id="T_3c8f4_row146_col0" class="data row146 col0" >block_groups.3.block_group.1.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_3c8f4_row146_col1" class="data row146 col1" >(512,)</td>
      <td id="T_3c8f4_row146_col2" class="data row146 col2" >512</td>
      <td id="T_3c8f4_row146_col3" class="data row146 col3" >512</td>
      <td id="T_3c8f4_row146_col4" class="data row146 col4" >(512,)</td>
      <td id="T_3c8f4_row146_col5" class="data row146 col5" >layer4.1.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row147" class="row_heading level0 row147" >147</th>
      <td id="T_3c8f4_row147_col0" class="data row147 col0" >block_groups.3.block_group.1.left.conv_block.2.conv.weight</td>
      <td id="T_3c8f4_row147_col1" class="data row147 col1" >(2048, 512, 1, 1)</td>
      <td id="T_3c8f4_row147_col2" class="data row147 col2" >1048576</td>
      <td id="T_3c8f4_row147_col3" class="data row147 col3" >1048576</td>
      <td id="T_3c8f4_row147_col4" class="data row147 col4" >(2048, 512, 1, 1)</td>
      <td id="T_3c8f4_row147_col5" class="data row147 col5" >layer4.1.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row148" class="row_heading level0 row148" >148</th>
      <td id="T_3c8f4_row148_col0" class="data row148 col0" >block_groups.3.block_group.1.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_3c8f4_row148_col1" class="data row148 col1" >(2048,)</td>
      <td id="T_3c8f4_row148_col2" class="data row148 col2" >2048</td>
      <td id="T_3c8f4_row148_col3" class="data row148 col3" >2048</td>
      <td id="T_3c8f4_row148_col4" class="data row148 col4" >(2048,)</td>
      <td id="T_3c8f4_row148_col5" class="data row148 col5" >layer4.1.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row149" class="row_heading level0 row149" >149</th>
      <td id="T_3c8f4_row149_col0" class="data row149 col0" >block_groups.3.block_group.1.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_3c8f4_row149_col1" class="data row149 col1" >(2048,)</td>
      <td id="T_3c8f4_row149_col2" class="data row149 col2" >2048</td>
      <td id="T_3c8f4_row149_col3" class="data row149 col3" >2048</td>
      <td id="T_3c8f4_row149_col4" class="data row149 col4" >(2048,)</td>
      <td id="T_3c8f4_row149_col5" class="data row149 col5" >layer4.1.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row150" class="row_heading level0 row150" >150</th>
      <td id="T_3c8f4_row150_col0" class="data row150 col0" >block_groups.3.block_group.2.left.conv_block.0.conv.weight</td>
      <td id="T_3c8f4_row150_col1" class="data row150 col1" >(512, 2048, 1, 1)</td>
      <td id="T_3c8f4_row150_col2" class="data row150 col2" >1048576</td>
      <td id="T_3c8f4_row150_col3" class="data row150 col3" >1048576</td>
      <td id="T_3c8f4_row150_col4" class="data row150 col4" >(512, 2048, 1, 1)</td>
      <td id="T_3c8f4_row150_col5" class="data row150 col5" >layer4.2.conv1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row151" class="row_heading level0 row151" >151</th>
      <td id="T_3c8f4_row151_col0" class="data row151 col0" >block_groups.3.block_group.2.left.conv_block.0.batchnorm2d.weight</td>
      <td id="T_3c8f4_row151_col1" class="data row151 col1" >(512,)</td>
      <td id="T_3c8f4_row151_col2" class="data row151 col2" >512</td>
      <td id="T_3c8f4_row151_col3" class="data row151 col3" >512</td>
      <td id="T_3c8f4_row151_col4" class="data row151 col4" >(512,)</td>
      <td id="T_3c8f4_row151_col5" class="data row151 col5" >layer4.2.bn1.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row152" class="row_heading level0 row152" >152</th>
      <td id="T_3c8f4_row152_col0" class="data row152 col0" >block_groups.3.block_group.2.left.conv_block.0.batchnorm2d.bias</td>
      <td id="T_3c8f4_row152_col1" class="data row152 col1" >(512,)</td>
      <td id="T_3c8f4_row152_col2" class="data row152 col2" >512</td>
      <td id="T_3c8f4_row152_col3" class="data row152 col3" >512</td>
      <td id="T_3c8f4_row152_col4" class="data row152 col4" >(512,)</td>
      <td id="T_3c8f4_row152_col5" class="data row152 col5" >layer4.2.bn1.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row153" class="row_heading level0 row153" >153</th>
      <td id="T_3c8f4_row153_col0" class="data row153 col0" >block_groups.3.block_group.2.left.conv_block.1.conv.weight</td>
      <td id="T_3c8f4_row153_col1" class="data row153 col1" >(512, 512, 3, 3)</td>
      <td id="T_3c8f4_row153_col2" class="data row153 col2" >2359296</td>
      <td id="T_3c8f4_row153_col3" class="data row153 col3" >2359296</td>
      <td id="T_3c8f4_row153_col4" class="data row153 col4" >(512, 512, 3, 3)</td>
      <td id="T_3c8f4_row153_col5" class="data row153 col5" >layer4.2.conv2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row154" class="row_heading level0 row154" >154</th>
      <td id="T_3c8f4_row154_col0" class="data row154 col0" >block_groups.3.block_group.2.left.conv_block.1.batchnorm2d.weight</td>
      <td id="T_3c8f4_row154_col1" class="data row154 col1" >(512,)</td>
      <td id="T_3c8f4_row154_col2" class="data row154 col2" >512</td>
      <td id="T_3c8f4_row154_col3" class="data row154 col3" >512</td>
      <td id="T_3c8f4_row154_col4" class="data row154 col4" >(512,)</td>
      <td id="T_3c8f4_row154_col5" class="data row154 col5" >layer4.2.bn2.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row155" class="row_heading level0 row155" >155</th>
      <td id="T_3c8f4_row155_col0" class="data row155 col0" >block_groups.3.block_group.2.left.conv_block.1.batchnorm2d.bias</td>
      <td id="T_3c8f4_row155_col1" class="data row155 col1" >(512,)</td>
      <td id="T_3c8f4_row155_col2" class="data row155 col2" >512</td>
      <td id="T_3c8f4_row155_col3" class="data row155 col3" >512</td>
      <td id="T_3c8f4_row155_col4" class="data row155 col4" >(512,)</td>
      <td id="T_3c8f4_row155_col5" class="data row155 col5" >layer4.2.bn2.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row156" class="row_heading level0 row156" >156</th>
      <td id="T_3c8f4_row156_col0" class="data row156 col0" >block_groups.3.block_group.2.left.conv_block.2.conv.weight</td>
      <td id="T_3c8f4_row156_col1" class="data row156 col1" >(2048, 512, 1, 1)</td>
      <td id="T_3c8f4_row156_col2" class="data row156 col2" >1048576</td>
      <td id="T_3c8f4_row156_col3" class="data row156 col3" >1048576</td>
      <td id="T_3c8f4_row156_col4" class="data row156 col4" >(2048, 512, 1, 1)</td>
      <td id="T_3c8f4_row156_col5" class="data row156 col5" >layer4.2.conv3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row157" class="row_heading level0 row157" >157</th>
      <td id="T_3c8f4_row157_col0" class="data row157 col0" >block_groups.3.block_group.2.left.conv_block.2.batchnorm2d.weight</td>
      <td id="T_3c8f4_row157_col1" class="data row157 col1" >(2048,)</td>
      <td id="T_3c8f4_row157_col2" class="data row157 col2" >2048</td>
      <td id="T_3c8f4_row157_col3" class="data row157 col3" >2048</td>
      <td id="T_3c8f4_row157_col4" class="data row157 col4" >(2048,)</td>
      <td id="T_3c8f4_row157_col5" class="data row157 col5" >layer4.2.bn3.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row158" class="row_heading level0 row158" >158</th>
      <td id="T_3c8f4_row158_col0" class="data row158 col0" >block_groups.3.block_group.2.left.conv_block.2.batchnorm2d.bias</td>
      <td id="T_3c8f4_row158_col1" class="data row158 col1" >(2048,)</td>
      <td id="T_3c8f4_row158_col2" class="data row158 col2" >2048</td>
      <td id="T_3c8f4_row158_col3" class="data row158 col3" >2048</td>
      <td id="T_3c8f4_row158_col4" class="data row158 col4" >(2048,)</td>
      <td id="T_3c8f4_row158_col5" class="data row158 col5" >layer4.2.bn3.bias</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row159" class="row_heading level0 row159" >159</th>
      <td id="T_3c8f4_row159_col0" class="data row159 col0" >output_layers.2.weight</td>
      <td id="T_3c8f4_row159_col1" class="data row159 col1" >(1000, 2048)</td>
      <td id="T_3c8f4_row159_col2" class="data row159 col2" >2048000</td>
      <td id="T_3c8f4_row159_col3" class="data row159 col3" >2048000</td>
      <td id="T_3c8f4_row159_col4" class="data row159 col4" >(1000, 2048)</td>
      <td id="T_3c8f4_row159_col5" class="data row159 col5" >fc.weight</td>
    </tr>
    <tr>
      <th id="T_3c8f4_level0_row160" class="row_heading level0 row160" >160</th>
      <td id="T_3c8f4_row160_col0" class="data row160 col0" >output_layers.2.bias</td>
      <td id="T_3c8f4_row160_col1" class="data row160 col1" >(1000,)</td>
      <td id="T_3c8f4_row160_col2" class="data row160 col2" >1000</td>
      <td id="T_3c8f4_row160_col3" class="data row160 col3" >1000</td>
      <td id="T_3c8f4_row160_col4" class="data row160 col4" >(1000,)</td>
      <td id="T_3c8f4_row160_col5" class="data row160 col5" >fc.bias</td>
    </tr>
  </tbody>
</table>




```python
compare_predictions(my_resnet50, pretrained_resnet50, IMAGE_FILENAMES, atol=1e-5)
```

    Models are equivalent!


### Convolutions and MaxPool from Scratch

We'll take a look under the hood and see how some very low-level operations are performed using `torch.as_strided()`. This is not something that we'll need for implementing various architectures in the future - we can simply use PyTorch modules or write our own at the abstraction level used in the previous sections. However, this is useful for getting a much better understanding of how these operations work.

Here is some useful reading to get an understanding of how `torch.as_strided()` works:
* [Using .as_strided for creating views of NumPy arrays](https://www.youtube.com/watch?v=VlkzN00P0Bc)
* [as_strided and sum are all you need (...to implement the non-pointwise operations in a neural network)](https://jott.live/markdown/as_strided)

The key thing to understand is the underlying representation of tensors is that the tensors do not take on their shapes in memory. Rather, they live in 1D contiguous arrays (or non-contiguous if the tensor was created by striding over a continuous array).

First we'll implement some simple matrix operations to get warmed up to `torch.as_strided`. Then we'll build up to the full `conv2d` and `maxpool2d`. This follows [ARENA 3.0 material](https://arena3-chapter0-fundamentals.streamlit.app/[0.2]_CNNs_&_ResNets) very closely.

##### Warm-up: Matrix operations

###### trace

Let's assume this matrix lives in contiguous memory.


```python
def as_strided_trace(mat: Float[Tensor, "i j"]) -> Float[Tensor, ""]:
  '''
  Similar to `torch.trace`, using only 'as_strided` and `sum` methods.
  '''
  assert mat.shape[0] == mat.shape[1]
  len = mat.shape[0]
  return mat.as_strided((1, len), (1, len + 1)).sum(1)
```

###### matrix-vector multiplication


```python
def as_strided_mv(mat: Float[Tensor, "i j"], vec: Float[Tensor, "j"]) -> Float[Tensor, "i"]:
  '''
  Similar to `torch.matmul`, using only `as_strided` and `sum` methods.
  '''
  return (mat * vec.as_strided(mat.shape, (0,vec.stride()[0]))).sum(1)
```

###### matrix-matrix multiplication


```python
def as_strided_mm(matA: Float[Tensor, "i j"], matB: Float[Tensor, "j k"]) -> Float[Tensor, "i k"]:
  '''
  Similar to `torch.matmul`, using only `as_strided` and `sum` methods.
  '''
  assert(matA.shape[1] == matB.shape[0])
  i,j = matA.shape
  j,k = matB.shape
  A = matA.as_strided((i,j,k), (matA.stride(0), matA.stride(1), 0))
  B = matB.as_strided((i,j,k), (0, matB.stride(0), matB.stride(1)))
  return (A*B).sum(1)
```

#### Building up to full conv2d and maxpool2d

###### extra-minimal conv1d

Here we'll implement `conv1d` with `padding=0` and `stride=1`, and with batch size, number of input features, and number of outputs features all equal to 1.


```python
def conv1d_minimal_simple(x: Float[Tensor, "w"], weights: Float[Tensor, "kw"]) -> Float[Tensor, "ow"]:
    '''
    Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    Simplifications: batch = input channels = output channels = 1.

    x: shape (width,)
    weights: shape (kernel_width,)

    Returns: shape (output_width,)
    '''
    input_width = x.shape[0]
    kernel_width = weights.shape[0]
    output_width = input_width - kernel_width + 1
    new_size = (output_width, kernel_width)
    new_stride = (x.stride(0), x.stride(0))
    input = x.as_strided(size=new_size, stride=new_stride)
    return einops.einsum(input, weights, 'output_width kernel_width, kernel_width -> output_width')
```

###### minimal `conv1d` and `conv2d`

Here we'll implement `conv1d` and `conv2d` with `padding=0` and `stride=1`, but implements the "full version" for the other dimensions.


```python
def conv1d_minimal(x: Float[Tensor, "b ic w"], weights: Float[Tensor, "oc ic kw"]) -> Float[Tensor, "b oc ow"]:
    '''
    Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    b, ic, w = x.shape
    oc, ic2, kw = weights.shape
    assert(ic == ic2)
    ow = w - kw + 1
    new_size = (b, ic, ow, kw)
    new_stride = (x.stride(0), x.stride(1), x.stride(2), x.stride(2))
    input = x.as_strided(size=new_size, stride=new_stride)
    return einops.einsum(input, weights, 'b ic ow kw, oc ic kw -> b oc ow')
```


```python
def conv2d_minimal(x: Float[Tensor, "b ic h w"], weights: Float[Tensor, "oc ic kh kw"]) -> Float[Tensor, "b oc oh ow"]:
    '''
    Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    b, ic, h, w = x.shape
    oc, ic2, kh, kw = weights.shape
    assert(ic == ic2)
    oh = h - kh + 1
    ow = w - kw + 1
    new_size = (b, ic, oh, ow, kh, kw)
    new_stride = (x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(2), x.stride(3))
    input = x.as_strided(size=new_size, stride=new_stride)
    return einops.einsum(input, weights, 'b ic oh ow kh kw, oc ic kh kw -> b oc oh ow')
```

###### full `conv1d` and `conv2d`


```python
def conv1d(
    x: Float[Tensor, "b ic w"],
    weights: Float[Tensor, "oc ic kw"],
    stride: int = 1,
    padding: int = 0
) -> Float[Tensor, "b oc ow"]:
    '''
    Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    b, ic, w = x.shape
    oc, ic2, kw = weights.shape
    assert(ic == ic2)

    padded_width = w+padding*2
    padded_x = x.new_full(size=(b, ic, padded_width), fill_value=0)
    padded_x[..., padding:w+padding] = x

    ow = int((padded_width - kw)/stride) + 1
    new_size = (b, ic, ow, kw)

    b_s, ic_s, w_s = padded_x.stride()
    new_stride = (b_s, ic_s, w_s*stride, w_s)

    strided_padded_x = padded_x.as_strided(size=new_size, stride=new_stride)
    return einops.einsum(strided_padded_x, weights, 'b ic ow kw, oc ic kw -> b oc ow')
```

We'll implement a `conv2d` that allows for non-square kernels. We'll need to define a new type that can handle that flexibility, in addition to non-symmetrical padding.


```python
IntOrPair = Union[int, Tuple[int, int]]
Pair = Tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    '''Convert v to a pair of int, if it isn't already.'''
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)

# Examples of how this function can be used:

for v in [(1, 2), 2, (1, 2, 3)]:
    try:
        print(f"{v!r:9} -> {force_pair(v)!r}")
    except ValueError:
        print(f"{v!r:9} -> ValueError")
```

    (1, 2)    -> (1, 2)
    2         -> (2, 2)
    (1, 2, 3) -> ValueError


Now we can implement the full `conv2d`.


```python
def conv2d(
    x: Float[Tensor, "b ic h w"],
    weights: Float[Tensor, "oc ic kh kw"],
    stride: IntOrPair = 1,
    padding: IntOrPair = 0
) -> Float[Tensor, "b oc oh ow"]:
    '''
    Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    b, ic, h, w = x.shape
    oc, ic2, kh, kw = weights.shape
    assert(ic == ic2)

    pad_h, pad_w = force_pair(padding)
    padded_height = h+pad_h*2
    padded_width = w+pad_w*2
    padded_x = x.new_full(size=(b, ic, padded_height, padded_width), fill_value=0)
    padded_x[..., pad_h:h+pad_h, pad_w:w+pad_w] = x

    stride_h, stride_w = force_pair(stride)
    oh = int((padded_height - kh)/stride_h) + 1
    ow = int((padded_width - kw)/stride_w) + 1
    new_size = (b, ic, oh, ow, kh, kw)

    b_s, ic_s, h_s, w_s = padded_x.stride()
    new_stride = (b_s, ic_s, h_s*stride_h, w_s*stride_w, h_s, w_s)

    strided_padded_x = padded_x.as_strided(size=new_size, stride=new_stride)
    return einops.einsum(strided_padded_x, weights, 'b ic oh ow kh kw, oc ic kh kw -> b oc oh ow')
```

###### full maxpool2d

`maxpool2d` is very similar to `conv2d`, except the operations at the end. Max pooling is very similar to convolution in that we slide a window across a matrix, except instead of multiplying by the kernel and summing, we simply take the maximum in the window. Also, instead of having each output channel be a function of all the input channels, we have the same number of output channels as input channels since the max pooling operation is taken independently for each channel.


```python
def maxpool2d(
    x: Float[Tensor, "b ic h w"],
    kernel_size: IntOrPair,
    stride: Optional[IntOrPair] = None,
    padding: IntOrPair = 0
) -> Float[Tensor, "b ic oh ow"]:
    '''
    Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, output_height, output_width)
    '''
    b, ic, h, w = x.shape
    kernel_h, kernel_w = force_pair(kernel_size)
    pad_h, pad_w = force_pair(padding)
    if(stride):
      stride_h, stride_w = force_pair(stride)
    else:
      stride_h=kernel_h
      stride_w=kernel_w

    padded_h = h+pad_h*2
    padded_w = w+pad_w*2
    padded_size = (b, ic, padded_h, padded_w)
    padded_x = x.new_full(size=padded_size, fill_value=-t.inf)
    padded_x[..., pad_h:pad_h+h, pad_w:pad_w+w] = x

    oh = int((padded_h - kernel_h)/stride_h) + 1
    ow = int((padded_w - kernel_w)/stride_w) + 1
    new_size = (b, ic, oh, ow, kernel_h, kernel_w)

    b_s, ic_s, h_s, w_s = padded_x.stride()
    new_stride = (b_s, ic_s, h_s*stride_h, w_s*stride_w, h_s, w_s)

    strided_x = padded_x.as_strided(size=new_size, stride=new_stride)
    out = t.amax(strided_x, dim=(-1, -2))
    return out
```

#### Using our custom `conv2d` and `maxpool2d` in our custom modules


```python
CustomConv2d = CustomConv2dFactory(conv2d)
CustomMaxPool2d = CustomMaxPool2dFactory(maxpool2d)
```


```python
Conv2dLayer = Conv2dFactory(CustomConv2d, CustomBatchNorm2d, CustomReLU)
BottleneckConvBlock = ConvBlockFactory(Conv2dLayer, CustomSequential, bottleneck=True)
ResidualBlock = ResidualBlockFactory(BottleneckConvBlock, Conv2dLayer, CustomReLU)
BlockGroup = BlockGroupFactory(ResidualBlock, CustomSequential)
ResNet = ResNetFactory(Conv2dLayer, CustomMaxPool2d, BlockGroup, AveragePool, \
                       CustomFlatten, CustomLinear, CustomSequential)

my_resnet50 = ResNet(n_blocks_per_group=[3, 4, 6, 3],
                     middle_features_per_group=[64, 128, 256, 512],
                     out_features_per_group=[256, 512, 1024, 2048])
```


```python
my_resnet50 = copy_weights(my_resnet50, pretrained_resnet50)
compare_predictions(my_resnet50, pretrained_resnet50, IMAGE_FILENAMES, atol=1e-5)
```

    Models are equivalent!


Whew! This was a lot of code. But we've reached the end of implementing our own ResNet from scratch!

We followed the original paper for the high-level ResNet architectures (and upgraded to the "1.5 version" for the bottleneck design to match with PyTorch's architecture). We first made concrete models by using building blocks from PyTorch's `torch.nn` modules, then replaced them with our own modules by sub-classing `torch.nn.Module`. Then we went even deeper, using `.as_strided()` and `.sum()` to replace `torch.nn.functional` calls. Along the way, we tested the models by copying weights from PyTorch's pre-trained ResNet models and running them on some test images to see if we got the same results (modulo some tolerance due to how floating point operations were chained).

# Finetune for FashionMNIST

Now let's do some transfer learning. We should be able to take our pre-trained model (trained on ImageNet with RGB images and 1000 classes) and adapt it to be able to classify images from the FashionMNIST dataset (grayscale images and 10 classes).

We want to change the last layer of our model to output 10 classes instead of 1000 and freeze the weights of all the preceding layers.

For the input, since we want to deal with a single input channel rather than three, we could modify the first layer to take in only a single channel, but it'd be difficult to know which kernels for the three channels to discard. We could try to learn the first layer's weights from scratch, but let's try duplicating our single channel input into three channels instead.


```python
from torchvision.datasets import FashionMNIST
from torch.utils.data import Subset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
```

As part of adapting our model and data to play nice with each other, we need to resize each image from (1, 28, 28) to (3, 224, 224), the image shape that ResNet34 expects.


```python
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((224, 224)),
                                 transforms.Lambda(lambda x:
                                  einops.repeat(x, '1 h w -> 3 h w'))])
```


```python
train_dataset = FashionMNIST(root='fashion_mnist', train=True, download=True,
                             transform=transform)
test_dataset = FashionMNIST(root='fashion_mnist', train=False, download=True,
                            transform=transform)

train_data = Subset(train_dataset, t.arange(int(len(train_dataset)*0.8)))
valid_data = Subset(train_dataset, t.arange(int(len(train_dataset)*0.8),
                                            len(train_dataset)))
```

Let's visualize some of our transformed data.


```python
for i in range(10):
  plt.subplot(2, 5, i + 1)
  image = einops.rearrange(train_data[i][0], 'c h w -> h w c')
  plt.imshow(image, cmap=plt.get_cmap('gray'))
plt.subplots_adjust(wspace=0.6, bottom=0.4)
plt.show()
```

<img src="/assets/img/resnet/resnet_from_scratch_123_1.png">


Now, let's freeze the weights of our pre-trained model and change the output head to one that outputs 10 logits instead of 1000. We'll be training the weights of only the final layer.


```python
my_resnet34.requires_grad_(False)

my_resnet34.output_layers[-1] = CustomLinear(
  my_resnet34.out_features_per_group[-1], 10)
```

Let's reuse the `Learner` and `DataLoaders` classes we implemented in [Getting Hands On With ML: MNIST](https://henryjchang.github.io/hands-on-with-ml-mnist). We make some small updates to a) wrap iterating through the training data with `tqdm` since training will take a lot longer with a bigger model, b) move model and data to device, since we'll want to use a GPU if available, and c) add gradient accumulation just in case we run out of memory working with large batch sizes.


```python
device = t.device("cuda" if t.cuda.is_available() else "cpu")
```


```python
class Learner:
  def __init__(self, dataloaders, model, optimizer, loss_func, metric,
               scheduler=None, gradient_accumulation_batch_size=64):
    self.dataloaders = dataloaders
    self.model = model.to(device)
    self.optimizer = optimizer
    self.loss_func = loss_func
    self.metric = metric
    self.scheduler = scheduler
    self.val_losses = []
    self.gradient_accumulation_bs = gradient_accumulation_batch_size
    self.gradient_accumulation_count = 0

  def fit(self, epochs):
    for epoch in tqdm(range(epochs)):
      self.model.train()
      train_loss = 0.
      for (train_features, train_labels) in tqdm(self.dataloaders.train_dl()):
        train_features = train_features.to(device)
        train_labels = train_labels.to(device)
        preds = self.model(train_features)
        loss = self.loss_func(preds, train_labels)
        train_loss += loss
        loss.backward()
        self.optimizer.step()
        self.gradient_accumulation_count += len(train_labels)
        if self.gradient_accumulation_count - self.gradient_accumulation_bs > 0:
          self.gradient_accumulation_count = 0
          self.optimizer.zero_grad()
        if self.scheduler:
          self.scheduler.step()
      print("avg training loss: ", train_loss / len(self.dataloaders.train_dl()))

      self.model.eval()
      with t.no_grad():
        val_losses = 0.
        val_metric = 0.
        metric_results = []
        for (val_features, val_labels) in self.dataloaders.valid_dl():
          val_features = val_features.to(device)
          val_labels = val_labels.to(device)
          preds = self.model(val_features)
          val_losses += self.loss_func(preds, val_labels)
          val_metric += self.metric(preds, val_labels)
        num_batches = len(self.dataloaders.valid_dl())
        print("avg validation loss: ", val_losses / num_batches)
        print("metric: ", val_metric / num_batches)
```


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


```python
def accuracy(preds, labels):
  return (t.argmax(preds, axis=1) == labels).float().mean()
```


```python
bs = 64
train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True, drop_last=True)
valid_dataloader = DataLoader(valid_data, batch_size=bs, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=bs, drop_last=True)
dls = DataLoaders(train_dataloader, valid_dataloader)
```


```python
epochs = 3
optimizer = t.optim.Adam(my_resnet34.parameters(), lr=1e-3)
loss_func = t.nn.CrossEntropyLoss()
scheduler = t.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-2, steps_per_epoch=len(train_dataloader), epochs=epochs)
learner = Learner(
    dls, my_resnet34.to(device), optimizer, loss_func, accuracy, scheduler,
    gradient_accumulation_batch_size=64)
learner.fit(epochs)
```


      0%|          | 0/3 [00:00<?, ?it/s]



      0%|          | 0/750 [00:00<?, ?it/s]


    avg training loss:  tensor(0.9040, device='cuda:0', grad_fn=<DivBackward0>)
    avg validation loss:  tensor(1.2077, device='cuda:0')
    metric:  tensor(0.7381, device='cuda:0')



      0%|          | 0/750 [00:00<?, ?it/s]


    avg training loss:  tensor(0.7149, device='cuda:0', grad_fn=<DivBackward0>)
    avg validation loss:  tensor(0.4802, device='cuda:0')
    metric:  tensor(0.8446, device='cuda:0')



      0%|          | 0/750 [00:00<?, ?it/s]


    avg training loss:  tensor(0.4177, device='cuda:0', grad_fn=<DivBackward0>)
    avg validation loss:  tensor(0.3967, device='cuda:0')
    metric:  tensor(0.8632, device='cuda:0')


Cool! We've successfully finetuned a ResNet34 model, via feature extraction, to classify FashionMNIST images. We could probably achieve much better results if we didn't freeze the weights or if we trained a model from scratch (perhaps with a single input channel).

Hopefully you've gotten a much better understanding of what a ResNet is and how to work with them after reading this post.

# Other Resources

Other resources that I've found helpful in aiding my understanding of CNNs and ResNets are linked below:

https://github.com/fastai/fastbook/blob/master/13_convolutions.ipynb

https://github.com/fastai/course22p2/blob/master/nbs/13_resnet.ipynb

https://arena3-chapter0-fundamentals.streamlit.app/[0.2]_CNNs_&_ResNets

https://cs231n.github.io/convolutional-networks/

http://colah.github.io/posts/2014-07-Conv-Nets-Modular/

