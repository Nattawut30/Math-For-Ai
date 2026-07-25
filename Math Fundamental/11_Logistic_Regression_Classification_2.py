""" Logistic Regression Classification (Contd.) """

""" 1. R-Squared """

# Overfitting and varienace is still a problem to all ML models
# We can borrow and adapt several metrics from linear regression and apply them to logistic regression.
# Let's start with R^2 for a given logistic regression
# R^2 indicates how well a given independent variable explains a dependent variable

# R^2 = (log likelihood) - (log likelihood fit) / (log likelihood)
# We cannot use residuals here like in linear Regression but we can project the outcomes back pnto the logistic curve
# We will convert the "false" likelihoods by subtracting from 1.0

# 11.1: Calcilating the log likelihood of the fit
from math import log, exp
import pandas as pd

patient_data = pd.read_csv('https://bit.ly/33ebs2R', delimiter=",").itertuples()

b0 = -3.17576395
b1 = 0.69267212

def logistic_function(x):
    p = 1.0 / (1.0 + exp(-(b0 + b1 * x)))
    return p

# Sum the log-likelihood
log_likelihood_fit = 0.0

for p in patient_data:
    if p.y == 1.0:
        log_likelihood_fit += log(logistic_function(p.x))
    elif p.y == 0.0:
        log_likelihood_fit += log(1.0 - logistic_function(p.x))

print(log_likelihood_fit) # -9.94616 ....

# We can use some binary subtraction between the true and false cases to mathematrically eliminate one or the other
# In this case, we multiply by 0 and therefore apply either the true or the false case, but not both the sum accordingly

# 11.2: Consolidating our log likelihood logic into a single line
log_likelihood_fit = sum(log(logistic_function(p.x)) * p.y +
                         log(1.0 - logistic_function(p.x)) * (1.0 - p.y)
                         for p in patient_data)

# We need one more datapoint to calculate the R^2: the log likelihood that estimates without using any input variables and simply uses the number of true cases divided by all cases
# (effectively leaving only the intercept)

# 11.3: Log likelihood of patients
import pandas as pd
from math import log, exp

patient_data = list(pd.read_csv('https://bit.ly/33ebs2R', delimiter=",") \
                    .itertuples())

likelihood = sum(p.y for p in patient_data) / len(patient_data)

log_likelihood = 0.0

for p in patient_data:
    if p.y == 1.0:
        log_likelihood += log(likelihood)
    elif p.y == 0.0:
        log_likelihood += log(1.0 - likelihood)

print(log_likelihood) # -14.34107...

# Compress that "for loop" and "if" expression into a single line,
# using some binary multiplication logic to handle both true and false cases.

# 11.4: Consolidating the log likelihood into a single line
log_likelihood = sum(log(likelihood) * p.y + log(1.0 - likelihood) * (1.0 - p.y) \
                     for p in patient_data)

# R^2 = (log likelihood) - (log likelihood fit) / (log likelihood)

# 11.5: Calculating the R^2 for logistic Regression
import pandas as pd
from math import log, exp

patient_data = list(pd.read_csv('https://bit.ly/33ebs2R', delimiter=",") \
                    .itertuples())

# Declares fitted logistic regression
b0 = -3.17576395
b1 = 0.69267212

def logistic_function(x):
    p = 1.0 / (1.0 + exp(-(b0 + b1 * x)))
    return p

# Calculate the log likelihood of the fit
log_likelihood_fit = sum(log(logistic_function(p.x)) * p.y +
                         log(1.0 - logistic_function(p.x)) * (1.0 - p.y)
                         for p in patient_data)

# Calculate the log likelihood without fit
likelihood = sum(p.y for p in patient_data) / len(patient_data)

log_likelihood = sum(log(likelihood) * p.y + log(1.0 - likelihood) * (1.0 - p.y) \
                     for p in patient_data)

# Calculate R-Square
r2 = (log_likelihood - log_likelihood_fit) / log_likelihood

print(r2) # 0.30645....

# On Linear Regression, a poor fit will be closer to an R^2 of 0.0 and a greater fit will be closer to 1.0
# We can conclude that hours of exposure is mediocre for predicting symptoms as the R^2 is 0.30645
# This logistic Regression has a perfect R^2 of 1.0 because...
# There is a clearn divide in outcomes predicted by hours of exposure.

""" 2. P-Values """

# "Chi-Square distribution" annotated as X^2 distribution.
# It is contininous and used in several areas of statistics including this one!

# If we take each value in a standard normal distribution (mean of 0 and stadnard deviation of 1) and square it, 
# that will give us the X^2 distribution with one degree of freedom

# Since we have two parameters (Hours of exposure & Whether Symptoms were shown)
# Our degree of freedom will be 1 because 2-1 = 1

# X^2 = 2(log likelihood fit) - (log likelihood)

# p-value = chi(2((log likelihood fit) - (log likelihood))

# 11.6: Calculating a p-value for a given Logistic Regresion
import pandas as pd
from math import log, exp
from scipy.stats import chi2

patient_data = list(pd.read_csv('https://bit.ly/33ebs2R', delimiter=",").itertuples())

# Declare fitted Logistic Regression
b0 = -3.17576395
b1 = 0.69267212

def logistic_function(x):
    p = 1.0 / (1.0 + exp(-(b0 + b1 * x)))
    return p

# Calculate the log likelihood of the fit
log_likelihood_fit = sum(log(logistic_function(p.x)) * p.y + 
                         log(1.0 - logistic_function(p.x)) * (1.0 - p.y)
                         for p in patient_data)

# Calculate the log likelihood without fit
likelihood = sum(p.y for p in patient_data) / len(patient_data)

log_likelihood = sum(log(likelihood) * p.y + log(1.0 - likelihood) * (1.0 - p.y) \
                     for p in patient_data)

# Calculate p-value
chi2_input = 2 * (log_likelihood_fit - log_likelihood)
p_value = chi2.pdf(chi2_input, 1) # 1 degree of freedom (n - 1)

print(p_value)

# p-value of 0.00166
# threshold for signifinace is .05
# We can say this data is statistically significant and was not by random chance.