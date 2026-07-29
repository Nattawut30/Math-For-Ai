""" Chapter 07: Neural Networks (Contd.) """

# Do not forget to analyze your data to check for imbalance classes.
# If the data is extreamly imbalanced (as in 99% of the data is one class), then remember to use confusion matrices
# to track the false positives and false negatives

""" 1. Backpropagation """
# Figuring out how to change each of the weight and bias values
# The nodes in one layer feed their weights and biases into the next layer,
# Then applies another set of weights and biases. This creates an onion-like nesting we need to untangle, starting with the output layer

# C = (A2 - Y)^2
# A2 = sigmoid(Z2)
# Z2 = W2 * A1 + B2
# A1 = ReLu(Z1)
# Z1 = W1 * X + B1
# This is where the chain rule can help us figure out this impact

# 14.1: Calculating the derivative of the cost function with respect to A2
from sympy import *

A2, y = symbols('A2 Y')
C = (A2 - y) ** 2
dC_dA2 = diff(C, A2)
print(dC_dA2) # 2*A2 - 2*Y

# A2 is the output of an activation function = logistic function
# taking the derivative of a sigmoid curve

# 14.2: Finding the derivative of A2 with respect to Z2
from sympy import *

Z2 = symbols('Z2')

logistic = lambda x:1 / (1 + exp(-x))

A2 = logistic(Z2)
dA2_dZ2 = diff(A2, Z2)
print(dA2_dZ2) 

# The derivative of Z2 with respect to W2 is fine for A1
# Linear Function and going to return the slope

# 14.3: Derivative of Z2 with respect to W2
from sympy import *

A1, W2, B2 = symbols('A1, W2, B2')

Z2 = A1 * W2 + B2
dZ2_dw2 = diff(Z2, W2)
print(dZ2_dw2) # A1
# Don't get lost in the math!

# 14.4: Calculating all the partial derivatives we will need for our neural network
from sympy import *

W1, W2, B1, B2, A1, A2, Z1, Z2, X, Y =  symbols('W1 W2 B1 B2 A1 A2 Z1 Z2 X Y')

# Calculate derivative of cost function with respect to A2
C = (A2 - Y) ** 2
dC_dA2 = diff(C, A2)
print("dC_dA2 = ", dC_dA2) # 2 * A2 - 2 * Y

# Calculate derivative of A2 with repect to Z2
logistic = lambda x:1 / (1 + exp(-x))
_A2 = logistic(Z2)
dA2_dZ2 = diff(_A2, Z2)
print("dA2_dZ2 = ", dA2_dZ2) # exp(-Z2)/(1 + exp(-Z2))**2

# Calculate aerivative of Z2 with respect to A1
_Z2 = A1 * W2 + B2
dZ2_dA1 = diff(_Z2, A1)
print("dZ2_dA1 = ", dZ2_dA1 ) # W2

# Calculate derivative of Z2 with respect to W2
dZ2_dw2 = diff(_Z2, W2)
print("dZ2_dw2 = ", dZ2_dw2) # A1

# Calculate derivative of Z2 with respect to B2
dZ2_dB2 = diff(_Z2, B2)
print("dZ2_dB2 = ", dZ2_dB2) # 1

# Calculate derivative of A1 with respect to Z1
relu = lambda x: Max(x, -0)
_A1 = relu(Z1)
d_relu = lambda x: x > 0 # slope is 1 if positive, 0 if negative
dA1_dZ1 = d_relu(Z1)
print("dA1_dZ1 = ", dA1_dZ1) # Z1 > 0

# Calculate derivative of Z1 with respect to W1
_Z1 = X * W1 + B1
dZ1_dW1 = diff(_Z1, W1)
print("dZ1_dw1 = ", dZ1_dW1) # X

# Calcualte derivative of Z1 with respect to B1
dZ1_dB1 = diff(_Z1, B1)
print("dZ1_dB1 = ", dZ1_dB1) # 1

# Derivatives work with smooth curves, not jagged corners that exist on ReLu.
# These partial detivatives can be chained together to create new partial detivativs with respect to the weights and biases.

# Automatic Differentiation
# You can use JAX library made by Google. It is nearly identical to NumPy except it allows calculating derivatives on parameter packaged as matrices.

""" 2. Stochastic Gradient Descent """

# Batch and mini-batch gradient descent are commonly used in neural networks and deep learning.

# 14.5: Implementing a neural network using stochastic gradient descent
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

all_data = pd.read_csv("https://tinyurl.com/y2qmhfsr")

# Learning rate controls how slowly we approach a solution
# Make it too small, it will take too long to run
# Make it too big, it will likely overshoot and miss the solution.
L = 0.05

# Extract the input columns, scale down by 255
all_inputs = (all_data.iloc[:, 0:3].values / 255.0)
all_outputs = (all_data.iloc[:, -1].values)

# Split train and test data sets
X_train, X_test, Y_train, Y_test = train_test_split(all_inputs, all_outputs, test_size=1 / 3)
n = X_train.shape[0]

# Build Neural network with weights and biases
# With random initializatio
w_hidden = np.random.rand(3, 3)
w_output = np.random.rand(1, 3)

b_hidden = np.random.rand(3, 1)
b_output = np.random.rand(1, 1)

# Activation functions
relu = lambda x: np.maximum(x, 0)
logistic = lambda x: 1 / (1 + np.exp(-x))

# Runs inputs through the neural network to get predicted outputs
def forward_prop(X):
    Z1 = w_hidden @ X + b_hidden
    A1 = relu(Z1)
    Z2 = w_output @ A1 + b_output
    A2 = logistic(Z2)
    return Z1, A1, Z2, A2

# Derivatives of activation functions
d_relu = lambda x: x > 0
d_logistic = lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2

# returns slopes for weights and biases
# using chain rule
def backward_prop(Z1, A1, Z2, A2, X, Y):
    # Loss derivative for mean squared error (or half-MSE)
    dC_dA2 = 2 * (A2 - Y)
    dA2_dZ2 = d_logistic(Z2)
    
    # Delta for output layer (element-wise product)
    dZ2 = dC_dA2 * dA2_dZ2
    
    # Gradients for output layer
    dC_dW2 = dZ2 @ A1.T
    dC_dB2 = dZ2

    # Delta for hidden layer
    dC_dA1 = w_output.T @ dZ2
    dA1_dZ1 = d_relu(Z1)
    dZ1 = dC_dA1 * dA1_dZ1

    # Gradients for hidden layer
    dC_dW1 = dZ1 @ X.T
    dC_dB1 = dZ1

    return dC_dW1, dC_dB1, dC_dW2, dC_dB2

# Execute gradient descent
for i in range(100_000):
    # randomly select one of the training data
    idx = np.random.choice(n, 1, replace=False)

    X_sample = X_train[idx].transpose()
    Y_sample = Y_train[idx]

    # run randomly selected training data through neural network
    Z1, A1, Z2, A2 = forward_prop(X_sample)

    # distribute error through backpropagation
    # and return slopes for weights and biases
    dW1, dB1, dW2, dB2 = backward_prop(Z1, A1, Z2, A2, X_sample, Y_sample)

    # update weight and biases
    w_hidden -= L * dW1
    b_hidden -= L * dB1
    w_output -= L * dW2
    b_output -= L * dB2

# Calculate Accuracy
test_predictions = forward_prop(X_test.transpose())[3] # grab only A2
test_comparisons = np.equal((test_predictions >= .5).flatten().astype(int), Y_test)
accuracy = sum(test_comparisons.astype(int) / X_test.shape[0])
print("ACCURACY: ", accuracy) # 97% - 99%

# We perform 100,000 iterations of stochastic gradient descent.
# Splitting the training and testing data by 2/3 and 1/3, repectively, I should get around 97%-99%
# Accuracy depends on how the randomness works out.
# After training, my neural network correctly identified 97% -  99% of the testing data with the right light/dark font prediction

# The backward_prop() functions is key!
# implementing the chain rule to take the error in the output node and divide it up and distribute it backward to the output and hidden weights/biases to get the slopes with respect to weight/biases
# Then take those slopes and nudge the weights/biases in the for loop, 
# then multiplying with the learning rate L

# Matrix-Vector multiplication to distribute the error backward based on the slopes and we transpose matrices and vectors when needed
# So the dimensions between rows and columns match up

