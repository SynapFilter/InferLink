# InferLink
Inferring the fragility, robustness and antifragility of links and layers in deep neural networks using synaptic filters. We define frgaile links and layers

## The Algorithm

The Synaptic filtering algorithm is a signal processing tool to decipher fragile, robust and antifragile links and layers in deep neural networks. The synaptic filtering algorithm is applied to a network with $`l`$ nodes (layers) and $`\theta`$ vertices (links) that are perturbed using systematic ablation of the synaptic links. We apply discrete parameter thresholds using three different types of synaptic filters: (1) the optimal low-pass filter, (2) the optimal high-pass filter, and (3) the pulse wave filter. We use the response of networks to the different synaptic filters to characterise fragile, robust and antifragile parameters of the evaluated networks. We further evaluate the network responses to the synaptic filters against an adversarial attack and compare the adversarial responses to the network response to a clean dataset and in doing so, we highlight parameters targeted by the adversary.

## Dependencies and Configurations

The Synaptic filtering algorithm is written in Python version 3.7.13 on Google Colab notebooks and is compatible with TPU useage. The code contains the following dependencies:

* [Pytorch 1.11.0](https://pytorch.org/get-started/locally/)
* [XLA Pytorch TPU client 0.10](https://github.com/pytorch/xla)
* 


## Structure of repository

The structure of the repository is organised as follows:

```

Robustness_analysis
│   README.md
│   synaptic_filtering.ipynb
|   model_trainer.ipynb
|   plots
│
└───Robustness_analysis
│   └───Dataset
|   |   └───MNIST
|   |   |   └───processed
|   |   |   |   └───test.pt
|   |   |   |   └───training.pt
|   |   |   └───raw
|   |   |   |   └───t10k-images-idx3-ubyte
|   |   |   |   └───t10k-images-idx3-ubyte.gz
|   |   |   |   └───t10k-labels-idx3-ubyte
|   |   |   |   └───t10k-labels-idx3-ubyte.gz
|   |   |   |   └───train-images-idx3-ubyte
|   |   |   |   └───train-images-idx3-ubyte.gz
|   |   |   |   └───train-labels-idx3-ubyte
|   |   |   |   └───train-labels-idx3-ubyte.gz


|   |   └───CIFAR10
|   |   |   └───cifar-10-python.tar.gz
|   |   |   └───cifar-10-batches-py
|   |   |   |   └───batches.meta
|   |   |   |   └───data_batch_1
|   |   |   |   └───data_batch_2
|   |   |   |   └───data_batch_3
|   |   |   |   └───data_batch_4
|   |   |   |   └───data_batch_5
|   |   |   |   └───readme.html
|   |   |   |   └───test_batch

|   |   └───tiny-imagenet-200
|   |   |   └───complete_dataset
|   |   |   |   └───test_set.pkl
|   |   |   |   └───train_set.pkl
|   └───saved_models
|   |   └───ResNet18
|   |   |   └───MNIST
|   |   |   |   └───Init_1
|   |   |   |   └───...
|   |   |   |   └───Init_N
|   |   |   |   └───global_data
|   |   |   |   └───Step_minmax
|   |   |   |   └───Step_maxmin
|   |   |   |   └───Pulse_minmax

|   |   |   └───CIFAR10
|   |   |   └───ImageNet
|   |   |   |   └───



## Running code

To run the experiments, open the .ipynb from root directory and follow cell comments. 

## links to datasets (open source dataset)

MNIST - http://yann.lecun.com/exdb/mnist/
CIFAR10 - https://www.cs.toronto.edu/~kriz/cifar.html
ImageNet Tiny - https://www.kaggle.com/c/tiny-imagenet


## What are the files and what are their purpose

1) The networks first need to be trained and saved into different Init folders within Saved_model->Network_name->dataset_name->Init_i. the networks can eb trained using the model_trainer.ipynb notebook, cell instructions are provideed within the notebook.

2) After the networks have been trained and svaed within the correct directories, it ispossible to run the synaptic filtering expeirments from the Synaptic_filtering.ipynb notebook. Here there are options for global and local anlaysis.

## Dependencies

To train and run the synaptic filtering experimetns on all networks, the following dependencies are required:

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, AutoMinorLocator
from tqdm.notebook import tqdm
import copy
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pll
import torch_xla.utils.serialization as xser

import torchvision.transforms as transforms
import torchvision.datasets

import random
import os
import pickle


