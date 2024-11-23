# MNIST Digit Classification with CNN

This repository contains a PyTorch-based implementation of a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset. The model achieves approximately 98% accuracy on the test set.

## Project Overview

The MNIST dataset is a standard benchmark in machine learning for image classification tasks. This project demonstrates how to build, train, and evaluate a CNN to classify images of digits (0-9) with high accuracy.

## Key Features

- Uses PyTorch for model development.
- Employs data preprocessing and augmentation using torchvision transforms.
- Implements a simple CNN architecture with convolutional, pooling, and fully connected layers.
- Provides visualizations of training and test loss over epochs.
- Generates confusion matrices for detailed evaluation of model performance.

## Getting Started

Make sure you have the following installed:

- Python 3.7+
- PyTorch
- torchvision
- NumPy
- pandas
- scikit-learn
- seaborn
- Matplotlib

You can install the required packages using pip:

pip install torch torchvision numpy pandas scikit-learn seaborn matplotlib


# Cloning the Repository

Clone the repository to your local machine:

git clone <https://github.com/cpena-estrada/MNIST_Multiclass_Classification>

## Usage

Training the Model

1. Download the MNIST dataset and preprocess it:
   - Transform the images into tensors using torchvision.
   - Normalize the data for efficient training.

2. Train the model for a specified number of epochs (5 in this case).

3. Monitor the training and testing loss over epochs:
   - Use Matplotlib for a loss vs epoch plot.

# Model Architecture
```
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)

        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2)
        #second pass/iteration
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)


        #review data to flatten it out
        X = X.reshape(-1, 16*5*5)

        #fully conected layers
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)
```


# Evaluating the Model

- Compute overall accuracy on the test set.

- Generate a confusion matrix for both the training and test datasets.


# References


- PyTorch Documentation

- Torchvision Transforms

- MNIST Dataset


# License

This project is licensed under the MIT License

# Author

Cristian Peña
