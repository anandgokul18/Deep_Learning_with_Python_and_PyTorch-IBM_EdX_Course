# IBM-Deep-Learning-with-Python-and-PyTorch

This repo holds my solutions (in Python) for the programming assignments of the edX's Deep Learning with Python and PyTorch Course by IBM

Course URL: https://www.edx.org/course/deep-learning-with-python-and-pytorch

## Contents:

### Module 1:

##### 1) 1- 1D tensors.ipynb

Covers basics of Tensors (in 1-dimension), Data types, Indexing and Slicing, Basic Operations in Pytorch like addition, multiplication, dot product, broadcasting and plotting functions

##### 2) 2- 2D Tensors.ipynb

Covers examples of 2-dimension tensors, Tensor creation in 2D, Iindexing and Slicing in 2D and basic operations on 2D tensors

##### 3) 3- Derivatives.ipynb

Covers derivatives and how to find derivatives in Pytorch using .backward(), partial derivatives wrt different variables and a cool way to find derivatives wrt to entire function (&not being limited to Pytorch which takes derivatives of only scalar functions)

##### 4) 4- Simple Dataset.ipynb

Covers how to build a simple dataset object. We will define a dataset class and override the Python's getitem() and len() methods in our class for the objects.

##### 5) 5- using Transforms on MNIST dataset.ipynb

Covers using prebuilt datasets (MNIST) and also using some Transform operations on the dataset

### Module 2:

##### 1) 1- Making a prediction.ipynb

Covers Linear Regression in 1D and making a prediction for a given value of x .ie. finding yhat using 1. Pytorch Class 'Linear' and using 2. Custom Modules

##### 2) 2- Linear Regression (only one parameter).ipynb

Covers a lot of basics: assuming linear relationship between x and y, we are going to train a model. Covers Loss functions, Gradient Descent to find loss function, Cost fn: Mean Square Error and training parameters in Pytorch manually

##### 3) 3- Linear Regression- Training 2 Parameters.ipynb

Covers the case when both weight (.ie. slope) and bias have to be trained

##### 4) 4- Stochastic_gradient_descent_with dataloader.ipynb

Covers the need for Stochastic Gradient Descent, problem with Stochastic Gradient Descent and implementing Stochastic Gradient Descent in Pytorch

##### 5) 5- Mini-Batch Gradient Descent.ipynb

Implementing Mini-batch gradient descent in Pytorch (also includes batch GD and Stochastic GD for comparison)

##### 6) 6- PyTorch way.ipynb

This covers implementing the above code the Pytorch way using in-built Pytorch functions/methods for our loss function and optimizer

##### 7) 7- Training and Validation Datasets.ipynb

This covers details about why we need training and validation datasets. Also, tells us how to implement PyTorch code to find ideal hyperparameters (here: learning_rate) for the same training dataset using the validation dataset. Then, base on lowest loss on validation dataset, we pick one of the models finally

##### 8) 8- Early_stopping.ipynb

This covers the implementation of early stopping .ie. using the model for an epoch which has the lowest loss on validation data, instead of running it for maximum epochs. Also, covers saving and loading a model





<hr>

The course contents and code are provided by IBM under MIT License

