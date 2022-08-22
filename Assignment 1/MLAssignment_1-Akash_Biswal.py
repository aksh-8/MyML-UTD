#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 00:10:07 2022

@author: Akash Biswal (axb200166)
"""

# In[1]
# Provided as assignment source code
# The true function
def f_true(x):
  y = 6.0 * (np.sin(x + 2) + np.sin(2*x + 4))
  return y

# We can generate a synthetic data set, with Gaussian noise.

# In[2]
# random synthetic data is generated
# Provided as assignment source code

import numpy as np                       # For all our math needs
n = 750                                  # Number of data points
X = np.random.uniform(-7.5, 7.5, n)      # Training examples, in one dimension
e = np.random.normal(0.0, 5.0, n)        # Random Gaussian noise
y = f_true(X) + e                        # True labels with noise


# In[3]
# plotting the raw data along with the true function
# Provided as assignment source code

import matplotlib.pyplot as plt          # For all our plotting needs
plt.figure()

# Plot the data
plt.scatter(X, y, 12, marker='o')           

# Plot the true function, which is really "unknown"
x_true = np.arange(-7.5, 7.5, 0.05)
y_true = f_true(x_true)
plt.plot(x_true, y_true, marker='None', color='r')

# In[4]
# Splitting the generated data into train, validation and test data
# Provided as assignment source code

# scikit-learn has many tools and utilities for model selection
from sklearn.model_selection import train_test_split
tst_frac = 0.3  # Fraction of examples to sample for the test set
val_frac = 0.1  # Fraction of examples to sample for the validation set

# First, we use train_test_split to partition (X, y) into training and test sets
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)

# Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)

# Plot the three subsets
plt.figure()
plt.scatter(X_trn, y_trn, 12, marker='o', color='orange')
plt.scatter(X_val, y_val, 12, marker='o', color='green')
plt.scatter(X_tst, y_tst, 12, marker='o', color='blue')

# In[5]

# X float(n, ): univariate data
# d int: degree of polynomial  

# Function to compute a Vandermonde matrix of dimension d 
# Problem 1.a

def polynomial_transform(X, d):
 
    #Vandermonde matrix
    Phi = []
    for val in X:
        temp = []
        for k in range(d+1):
            temp.append(np.power(val,k))
        
        Phi.append(temp)
    
    #converting 2-d list to a numpy array    
    Phi = np.asarray(Phi)
    return Phi

# In[6]

# Phi float(n, d): transformed data
# y   float(n,  ): labels

# Function to learn the weights using Phi and labels y
# via ordinary least squares regression
# Problem 1.b

def train_model(Phi, y):
    
    #computing w using the given formula
    w = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ y
    return w

# In[7]

# Phi float(n, d): transformed data
# y   float(n,  ): labels
# w   float(d,  ): linear regression model

# Using Phi, labels and linear regression model w
# the model is evaluated using mean squared error
# Problem 1.c

def evaluate_model(Phi, y, w):
    
    #predicted vales
    y_pred = Phi @ w
    #squared error
    er = (y_pred - y)**2
    er_sum = 0
    for val in er:
        er_sum += val
    
    # mean squared error
    mse = er_sum/len(y)
    return mse      

# In[8]

# Training the model using train set, eva;uating performance using validation set
# and estimating accuracy using test set

# Provided as assignment source code


w = {}               # Dictionary to store all the trained models
validationErr = {}   # Validation error of the models
testErr = {}         # Test error of all the models

for d in range(3, 25, 3):  # Iterate over polynomial degree
    Phi_trn = polynomial_transform(X_trn, d)                 # Transform training data into d dimensions
    w[d] = train_model(Phi_trn, y_trn)                       # Learn model on training data
    
    Phi_val = polynomial_transform(X_val, d)                 # Transform validation data into d dimensions
    validationErr[d] = evaluate_model(Phi_val, y_val, w[d])  # Evaluate model on validation data
    
    Phi_tst = polynomial_transform(X_tst, d)           # Transform test data into d dimensions
    testErr[d] = evaluate_model(Phi_tst, y_tst, w[d])  # Evaluate model on test data

# Plot all the models
plt.figure()
plt.plot(validationErr.keys(), validationErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testErr.keys(), testErr.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Polynomial degree', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([2, 25, -20, 200])

# In[9]

# code to visualize each model

# Provided as assignment source code

plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for d in range(9, 25, 3):
    X_d = polynomial_transform(x_true, d)
    y_d = X_d @ w[d]
    plt.plot(x_true, y_d, marker='None', linewidth=2)

plt.legend(['true'] + list(range(9, 25, 3)))
plt.axis([-8, 8, -15, 15])

# In[10]

# X float(n, ): univariate data
# B float(n, ): basis functions
# gamma float : standard deviation / scaling of radial basis kernel

# Function to compute a Radial-Basis Kernel
# used to model non-liner regression
# Problem 2.a

import math

def radial_basis_transform(X, B, gamma=0.1):
    
    #radial-basis kernel
    Phi = []
    for val1 in X:
        temp = []
        for val2 in B:
            temp.append(math.exp(-1*gamma*((val1-val2)**2)))
        
        Phi.append(temp)

    #converting 2-d list to a numpy array    
    Phi = np.asarray(Phi)
    return Phi


# In[11]

# Phi float(n, d): transformed data
# y   float(n,  ): labels
# lam float      : regularization parameter

# Learning weights via ridge regression using labels, regularization parameter and RB kernel
# Problem 2.b

def train_ridge_model(Phi, y, lam):
    
    #computing w using the given formula
    w = np.linalg.inv(Phi.T @ Phi + lam*(np.array(np.identity(len(y))))) @ Phi.T @ y
    return w
    
# In[12]

# exploring trade-off between fit and complexity by varying lambda
# evaluating performance
# Problem 2.c

w2 = {}               # Dictionary to store all the trained models
validationErr2 = {}   # Validation error of the models
testErr2 = {}         # Test error of all the models

for l_i, l in enumerate([10**i for i in range(-3,4)]):  # Iterate over polynomial degree
    Phi_trn2 = radial_basis_transform(X_trn, X_trn, gamma=0.1)                 # Transform training data into d dimensions
    w2[l_i] = train_ridge_model(Phi_trn2, y_trn, lam=l)                       # Learn model on training data
    
    Phi_val2 = radial_basis_transform(X_val, X_trn, gamma=0.1)                 # Transform validation data into d dimensions
    validationErr2[l_i] = evaluate_model(Phi_val2, y_val, w2[l_i])  # Evaluate model on validation data
    
    Phi_tst2 = radial_basis_transform(X_tst, X_trn, gamma=0.1)           # Transform test data into d dimensions
    testErr2[l_i] = evaluate_model(Phi_tst2, y_tst, w2[l_i])  # Evaluate model on test data

# Plot all the models
plt.figure()
plt.plot(validationErr2.keys(), validationErr2.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testErr2.keys(), testErr2.values(), marker='s', linewidth=3, markersize=12)

plt.xlabel('Lamda', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)

xn = [i for i in range(0, len([10**i for i in range(-3,4)]))]
plt.xticks(xn, ('0.001', '0.01', '0.1', '1', '10','100','1000'), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
#plt.axis([2, 25, -20, 200])

# In[13]

# code to visualize each model

# Problem 2.d

plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for l_i, l in enumerate([10**i for i in range(-3,4)]):
    #using the radial basis fucntion
    X_d2 = radial_basis_transform(x_true, X_trn, gamma=0.1)
    y_d2 = X_d2 @ w2[l_i]
    plt.plot(x_true, y_d2, marker='None', linewidth=2)

plt.legend(['true'] + [10**i for i in range(-3,4)], loc = "lower right")
plt.axis([-8, 8, -15, 15])

###############end###############
















