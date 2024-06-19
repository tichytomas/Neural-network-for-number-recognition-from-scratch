# Number Recognition Neural Network
Welcome to the Number Recognition project! This README provides a brief overview of what the project does, the technologies it uses, and how to set it up and run.

## What It Does
This project implements a neural network from scratch to recognize handwritten digits from the MNIST dataset. The neural network learns to classify digits by training on the dataset and optimizing its weights and biases through gradient descent.

## How It Works
### Core Components
Neural Network Architecture: A feedforward neural network with multiple layers.
Activation Function: Uses the sigmoid activation function.
Gradient Descent: Optimizes the weights and biases using mini-batch gradient descent and backpropagation.
MNIST Dataset: Uses the MNIST dataset for training and testing the neural network.
### Technologies and Libraries
Python: Main programming language.
NumPy: For numerical operations.
Pickle and Gzip: For loading and processing the MNIST dataset.
### Setup and Running
#### Prerequisites
Make sure you have Python installed, then install NumPy if you haven't already:

pip install numpy

#### Running the Project
Clone the Repository:
git clone https://github.com/tichytomas/Neural-network-for-number-recognition-from-scratch

cd /Neural-network-for-number-recognition-from-scratch

Run the Training:
python main.py