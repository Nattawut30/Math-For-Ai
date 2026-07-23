""" Chapter 06: Logistic Regression and Classification """

# Logistic Regression
# A type of regression that predicts a probability of an outcome given one or more independent varaibles.

# Classification
# Predicting categories rather than real numbers as we did with linear regression.

# Discrete or representative of whole numbers, integers, or booleans (1/0, true/false)
# Logistic Regression is trained on an output variable that is discrete (a binary of 1 or 0) or a categorical number (whole number)
# Out as a continuous variable in the form of probability, that can be convered into discrete value with a threshold

# Logistic regression is easy to implement and fairly resilient against outliers and other data challenges.
# Offering more practicality and performance than othr types of supervised machine learning.

""" 1. Understanding Logistic Regression """

# There's a industrial accident and trying to understand the impact of chemical exporsure.
# 11 patients who were exposed for differing numbers of hours to this chemical
# Some shows symptoms = value of 1
# others have no shown symptoms = value of 0

# Plotting on the graph: patients showed symptoms (1) or not (0) over x hours of exposure.
# At what length of time do patients start showing symptoms? see at 4 hours.
# from a cursory analysis: There is nearly 0% probability a patient exposed for fewer than 4 hours will show symptom
# But there is 100% probability for greater than 4 hours. (immediate jump to showing symptoms at approximately 4 hours.)

# Gather more data so the middle of the range has a mix of patients showing symptoms.
# The probability of patients showing symptoms will increase with each hour of exposure.
# Use "logistic function or an S-Shaped curve" where the output variable is squeezed between 0 and 1

# So, there is no distinct cutoff but rather a gradual transition from 0% probability to 100% probability (0 and 1)
# Logistic Regression results in a curve indicating a probability of belonging to the true category.

""" 2. Performing a Logistic Regression """

# Logistic Function
# an S-shaped curve (sigmoide curve) that, for a given set of input variables, produces an output variable between 0 and 1
# the output is 0 and 1 it can be used to represent a probability
# This formula uses Euler's number e
# The x variable is the independent/input variable.
# Beta0 and Beta1 are the coefficients we need to solve for!

# Logistic Regression actually has a close relationship to linear regression
# This linear function in the exponent is known as the log-odds function,
# Logistic function produces S-shaped curve we need to putput a shifting probability across an x-value

# 10.1: The logistic function in Python for one independent variable.

import math

def predict_probability(x, b0, b1):
    p = 1.0 / (1.0 + math.exp(-(b0 + b1 * x)))
    return p

# 10.2: Using SymPy to plot a logistic function
from sympy import *
b0, b1, x = symbols('b0 b1 x')

p = 1.0 / (1.0 + exp(-(b0 + b1 * x)))

p = p.subs(b0, -2.823)
p = p.subs(b1, 0.620)
print(p)

plot(p)

# Linear Regression we can also extend logistic regresion to more than one input variable

""" 3. Fitting The Logistic Curve """

# The data can have any mix of decimal, integer, and binary variables, but the output variable must be binary (0 and 1)
# The out put variable will be between 0 and 1, resembing a probability

# We use maximum likelihood estimation, which, as the name suggests, maximizes the likelihood a given logistic curve would output the observed data.
# but no closed form equation like in linear regression

# Using SciPY!
# You can copy/paste your code and can then reuse in between models

# 10.3: Using a plain logistic regression in SciPy
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('https://bit.ly/33ebs2R', delimiter=",")

X = df.values[:, :-1]

Y = df.values[:, -1]

model = LogisticRegression(penalty=None)

model.fit(X, Y)

print(model.coef_.flatten())
print(model.intercept_.flatten())

# Making Predictions
# To make specific predictions, use the predict() and predict_prob() functions on the model object in SciPy
# The predict() function will predict a specific class (True 1.0 or False 1.0) while the predict_prob() will output probabilities for each class

# flatten() the coefficient and intercept, come out as multidimensional matrices but with one element.
# Flattenning means collapsing a matrix of numbers into lesser dimensions,
# particularly when there are fewer elements than there dimensions.

""" 4. Using Maximum Likelihood and Gradient Descent """

# Maximum Likelihood estimation (MLE): maximizes the likelihood a given logistic curve would output the observed data.
# We combine probabilities or likelihood of multiple events by multiplying them together.
# In this case, we are calculating the likelihood we would see all these points for a given logistic regression curve.

# We fetch each likelihood off the logistic regression curve above or below each point.

# 10.4: Calculating the joint likelihood of ovserving all the points for a given logistic regression
import math
import pandas as pd

patient_data = pd.read_csv('https://bit.ly//33ebs2R', delimiter=",").itertuples()

b0 = -3.17576395
b1 = 0.69267212

def logistic_fucntion(x):
    p = 1.0 / (1.0 + math.exp(-(b0 + b1 * x)))
    return p

joint_likelihood = 1.0

for p in patient_data:
    if p.y == 1.0:
        joint_likelihood *= logistic_fucntion(p.x)
    elif p.y == 0.0:
        joint_likelihood *= (1.0 - logistic_fucntion(p.x))

print(joint_likelihood)

# When any number is raised to exponent 0, it will result in 1.
# Therefore, wherher y is 1 or 0, it will cause the opposite condition on the other side to evaluate to 1 and have no effect in multiplication
# We cannot do derivatives on expression that use if, so this will be helpful.

# Floating Point Underflow, This means that as decimals get smaller and smaller,
# it can happen in multipllication,
# the computers runs into limitations keeping track of that many decimal places

# good news! You can take the log() of each decimal you are multiplying and add them together

# 10.5: Using logarithmic addition
joint_likelihood = 0.0

for p in patient_data:
    joint_likelihood += math.log(logistic_fucntion(p.x)) ** p.y * \
                                (1.0 - logistic_fucntion(p.x)) ** (1.0 - p.y)

joint_likelihood = math.exp(joint_likelihood)

# Using SymPy do the partial derivatives for us and then compile and use them for gradient descent
# Nevertheless, We are trying to maximize rather than minimize, we add each adjustment to B0 and B1 rather than subtract like in least squares.

""" 5. Multivariable Logistic Regression """