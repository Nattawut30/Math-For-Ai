""" Chapter 07: Neural Networks (Contd.) """

# 15.1: Adding an interactive shell to our neurl network
# Interact and test with new colors
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

def predict_probability(r, g, b):
    X = np.array([[r, g, b]]).transpose() / 255
    Z1, A1, Z2, A2 = forward_prop(X)
    return A2

def predict_font_shade(r, g, b):
    output_values = predict_probability(r, g, b)
    if output_values > .5:
        return "DARK"
    else:
        return "LIGHT"

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

while True:
    col_input = input("Predict light or dark font. Input values R, G, B: ")
    if col_input in ['exit', 'q']: break

    (r, g, b) = col_input.split(",")
    print(predict_font_shade(int(r), int(g), int(b)))

# completed

""" Using Scikit-Learn """

# If you want to learn PyTorch and TensorFlow, consider a strong GPU.\
# Scikit-Learn = MLPClassifier = Multi-layer-perception classifier
# The neural network designed for classification, and it uses a logsitic output activation by default

# 15.2: Using Scikit-Learn neural network classifier
import pandas as pd

# Load data
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('https://bit.ly/3GsNzGt', delimiter=",")

# Extract input variables (all rows, all columns but last column)
# We should do some linear scaling here
X = (df.values[:, :-1] / 255.0)

# Extract output column (all rows, last column)
Y = df.values[:, -1]

# Separate training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3)

neural_networks = MLPClassifier(solver='sgd',
                                hidden_layer_sizes=(3, ),
                                activation='relu',
                                max_iter=100_000,
                                learning_rate_init=.05)

neural_networks.fit(X_train, Y_train)

# Check weights and biases
print(neural_networks.coefs_)
print(neural_networks.intercepts_)

print()

print("Training set score: %f" % neural_networks.score(X_train, Y_train))
print("Test set score: %f" % neural_networks.score(X_test, Y_test))
# I got 100% Accuracy once by the way....

""" Limitations of Neural Networks and Deep Learning """

# Flexibility with layers, nodes, and activation functions makes it flexible fitting to data in a nonlinear manner .... too flexible!
# It can overfit to the data

# The ML models has to be trained on countless combination of conditions around
# There is NO effective way to capture every type of event that is encountered on having more weights and biases in a neural network
# The ML was limited to label-making activity

# In reinforement learning environment like these that simulations are closed worlds, where infinite amounts of labeled data can be generated and learned throuh a virtual finite world
# The real world is not a simulation where we can generate unlimited amounts of data!
# The real world is full of infinite unpredicability and rare events.

# Thease ar all reasons why AI research loves to use board game and video games,
# because unlimited lebeled data can be generated easily and clearnly,

# ML, DL all work narrowly on defined problems... They broadly reason or choose their own tasks, or ponder objects they have not seen before.
# Neural Networks and Deep Learning do only what thet were programmed to do
# But they will not lose the usefulness!

# It's important to always consider what problem you are striving to solve without making a specific tool your primary objective.
# The use of deep learning has to be strategic and warranted.

# ***** There is much more power in the simplicity of pairing the right tool to the right problem! *****