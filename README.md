# InferLink
Inferring Significant Links and Layers of Deep Neural Networks to Adversarial Attacks Using Synaptic Filtering

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


│   
└───folder2
    │   file021.txt
    │   file022.txt
```

## Running code

To run the experiments, open the .ipynb from root directory and follow cell comments. 

## links to datasets (open source dataset)

## What are the files and what are their purpose -> the order in which the notebooks should be ran  

## Dependencies -> showing the colab TPU dependenices 

