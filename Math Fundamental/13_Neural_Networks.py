""" Chapter 07: Neural Networks """

# A neuural network is a multilayered regression containing layers of weights, biases, and nonlinear functions
# that reside between input variables and output variables.
# Deep Learning is a popular variant of neural networks that utilizes multiple "hidden" (or middle) layers of nodes containing weights and biases.
# Each node resembles a linear function before being passed to a nonlinear function called "an activation function"

# Neural networks offer exciting solution to problems previously difficult for computers to solve.
# From identifying objects in images to processing words in audio,
# Neural networks have created tools that affect our everyday lives.

# We will build a simple neural network in NumPy and use Scikit-Learn as a library implementation!

""" 1. When to Use Neural Networks and Deep Learning """

# "When all you have is a hammer, everything starts to look like a nail"
# Think of structured data as data that is easily represented as a table,
# with rows and columms.
# But perceptual problems like image classification are much less structured,
# as we are trying to find fuzzy correlations between group of pixels to identify shapes and pattern, not rows of data in a table!

# try to predict the next 4-5 words from audio are also perceptual problems
# Example of this using is call "Natural Language Processing (NLP)"

# Variants of Neural Networks
# - Convolutional Neural Networks = used for image recognition.
# - Long-Short-Term-Memory (LSTM) is used for predicting time-series or forecasting.
# - Recurrent neural network are often used for text-to-speech applications.

# *** You learn about the strength and limitation of the technique rather than be distracted by large datasets!
# *** Try not to use neural networks where simpler models will be more practical.

""" 2. A Simple Neural Network """

# In Computer Science, one way to represent a color is with RGB values (Red, Green, Blue)
# Each of these values is between 0 and 255 and express by 3 colors are mixed to create a desire values and colors.

# In ML perspective, We have 3 numeric input variables: Red, Green, Blue to capture a given background color.
# We need to fit a function to these input variables and output where a light(1) or dark(0) for the font should be used for the bg color

# "Mystery Math" a process that takes inputs and produces outputs. We have 3 numeric input variables R, G, B which are processed by this mystery math.
# (R), (G), (B) -> Mystery math -> Prediction
# This prediction output exprssed a probability!

# (R:255), (G:192), (B:203) -> Mystery Math -> Prediction(0.89)
# The mystery math suggest a light font because the output probability 0.89 is greater than 0.5!
# In the hidden (middle) layer notice that we produce 3 nodes, or functions of weight and biases, between the inputs and outputs.

# The output node repeats the same operation, taking the resulting weighted and summed outputs from the hidden layer and making them inputs into the final layer,
# where another set of weights and biases will be applied!

# This is a Regression just like Linear or Logistic Regression! But with many more parameters to solve for.
# The weight and bias values are analogous to the m and b or B1 and B0, parameters in a linear regression.

""" 3. Activation Functions """

# An activation function is a non-linear function that transform or compresses the weighted and summed values in a nodes,
# Helping the neural networks separate the data effectively so it can be classified!

# If you do not have the activation functions, your hidden layers will not be productive and will perform no better than a linear regression

# The ReLu activation function will zero out any negative outputs from the hidden nodes.
# If the weights, biases, and inputs multiply and sum to a negative number, it will be converted to 0.
# Otherwise, the output is left alone.

# 13.1: Plotting the Relu function
from sympy import *

x = symbols('x')
relu = Max(0, x)
plot(relu)

# ReLu = "rectified linear unit" or "Turn negative values into 0"
# ReLu has gotten popular for hidden layers in neural networks and DL becuase of its speed and mitigation of the vanishing gradient problem.

# The output layer has an important job: it take the piles of math from the hidden layers of the neural network and turns them into an interpretable result,
# such as presenting classification predictions.

# The output layer for this particular neural network uses the "logistic activation function" which is a simple sigmoid curve.
# So, the logistic regression is acting as a layer in out neural network.

# 13.2: Logistics Activation Frunction in SymPy
from sympy import *

x = symbols('x')
logistics = 1 / (1 + exp(-x))
plot(logistics)

# When we pass a node's weighted, biased, and summed value through an activation function, we now call that an "activated output"
# It has been filterd through the activation function.

# The activation function could have strengthened, weakened, or left the signal as is,
# This is where the brain and synapse metaphor for neural networks comes from

# Whether or not a font should be light(1) or dark(0). 
# If you wanted to support multiple classes, you could add more output nodes for each class
# You might consider using softmax as the output activation when you have multiple classes as well

# If you are trying to recognize handwritten digits 0-9, there would be 10 output nodes representing the probability a given image is each of those numbers!
# an output layer with 10 nodes presenting probabilities for 10 clasess.

# But I Don't Know What Activation Function To Use!
# 1. If you are unsure that activation to use, current best rpactices gravitate toward ReLu for middle layers and Logistic Sigmoid for output layer.
# 2. If you have multiple classification in the outputs, ue softmax for the output layer.

""" 4. Forward Propagation """

# Feed forward means we are simply inputting a color into the neutal network and seeing what it outputs

# 13.3: A simple forward propagation network with random weight and bias values
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

all_data = pd.read_csv("https://tinyurl.com/y2qmhfsr")

# Extract the input columns, scale down by 255
all_inputs = (all_data.iloc[:, 0:3].values / 255.0)
all_outputs = (all_data.iloc[:, -1].values)

# Split train and test data sets
X_train, X_test, Y_train, Y_test = train_test_split(all_inputs, all_outputs, test_size=1/3)
n = X_train.shape[0] # Numbers of training records

# Build neural network with weights and biases
# With random initialization
w_hidden = np.random.rand(3, 3)
w_output = np.random.rand(1, 3)

b_hidden = np.random.rand(3, 1)
b_output = np.random.rand(1, 1)

# Activation Functions
relu = lambda x: np.maximum(x, 0)
logistics = lambda x: 1 / (1 + np.exp(-x))

# Runs inputs through the neural network to get predicted outputs
def forward_prop(X):
    Z1 = w_hidden @ X + b_hidden
    A1 = relu(Z1)
    Z2 = w_output @ A1 + b_output
    A2 = logistics(Z2)
    return Z1, A1, Z2, A2

# Calculate Accuracy
test_predictions = forward_prop(X_test.transpose())[3] # grab only output layer 42
test_comparisons = np.equal((test_predictions >= .5).flatten().astype(int), Y_test)
accuracy = sum(test_comparisons.astype(int) / X_test.shape[0])
print("Accuracy: ", accuracy)

# These are declaing out weights and biases for both the hidden and output layers of out neural network
# Each node is represented as a row in a matrix. If ther are three nodes, there are three rows.
# If there is one node, there is one row. Each column holds a weight value for that node.
# Since there is one bias per node, there are going to be three rows of biases for the hidden layer and one row of biases for the output layer.

# 13.4: The activation functions and forward propagation function for our neural network
# Activation Functions
relu = lambda x: np.maximum(x, 0)
logistic = lambda x:1 / (1 + np.exp(-x))

# Runs inputs through the neural network to get predicted outputs
def forward_prop(X):
    Z1 = w_hidden @ X + b_hidden
    A1 = relu(Z1)
    Z2 = w_output @ A1 + b_output
    A2 = logistics(Z2)
    return Z1, A1, Z2, A2

# Thiscode is concisely executes our entire neural network using matrix multiplication and matrix-vector multiplication
# 1 and 2 indicate the operations belong to layer 1 and 2 respectively
# The "Z" indicates an unactivated output from the layer, and "A" is activated output from the layer.

# Z1 = W[Hidden]^X + B[Hidden]
# Just pass each value in that vector through the ReLu function and it will give us A1
# Because all the values are positive, it should not have an impact.

# A1 = ReLu(Z1)
# Let's take that hidden layer output A1 and pass it through the final layer to get Z2

# Z2 = W[Output]^A1 + B[Output]
# Pass this single value in Z2 through the activation function to get A2

# A2 = logistic(Z2)
# the final output that predict whether the bg color is light1 or dark0 font