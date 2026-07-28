""" Chapter 07: Neural Networks """

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

