""" Logistic Regression Classification (Contd.) """

""" 1. Train/Test Splits """

# Circle back 3-fold cross-validation:
# Data = Training -> Training -> Testing
# Data = Training -> Testing -> Training
# Data = Testing -> Training -> Training

# 12.1: Performing a Logistic Regression with 3-fold cross-validation
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

# Load the data
df = pd.read_csv("https://tinyurl.com/y6r7qjrp", delimiter=",")

X = df.values[:, :-1]
Y = df.values[:, -1]

# "random state" is the random seed, which we fix to 7
kfold = KFold(n_splits=3, random_state=7, shuffle=True)
model = LogisticRegression(penalty=None)
results = cross_val_score(model, X, Y, cv=kfold)

print("Accuracy Mean: %.3f (stdev=%.3f)" % (results.mean(), results.std()))

""" 1. Confusion Metrices """

# Imagine a model obseved peopel with the name "Michael" quit their job.
# I have 10000 employees, including one named "Michael" and another named "Sam".
# Michael is wrongly predicted to quit, and it is Sam that ends up quitting
# The accuracy of the model: 98%

# Imbalanced data where the event of interest is rare,
# the accuracy metric is horrendously misleaning for classification problems
# If anyone, tries to sell you a classification system on claims of accuracy, ask for a confusion matrix

# A "Confusion Matrix" is a grid that breaks out the predictions against the actual outcomes showing the true positives/negatives and false positives/negatives
# Your job is to dice up that accuracy metric into more specific accuracy metrics targeting different parts of the confusion matrix
# the precision and the sensitivities = 0 means the model fails entirely prediction

# 12.2: Creating a confusion matrix for a testing dataset in SciPy
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://bit.ly/3cManTi', delimiter=",")

# Extract input variables (all rows, all columns but last column)
X = df.values[:, :-1]

# Extract output variables (all rows, last column)\
Y = df.values[:, -1]

model = LogisticRegression(solver='liblinear')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33,
                                                        random_state=10)

model.fit(X_train, Y_train)
prediction = model.predict(X_test)

# The confusion matrix evaluates accuracy within each category
# [[true-positives, false-negatives]
# [false-positives, true-nagatives]]
# The diagonal represents correct predictions,
# So we want those to be higher

matrix = confusion_matrix(y_true=Y_test, y_pred=prediction)
print(matrix)

""" 2. Bayes's Theorem and Classification """
# Use this to bring in outside information to further validate findinds on a confusion matrix
# A confusion matrix for a medical test identifying a disease
# While we are flipping a conditional probability, we do not have to use Baye's Theorem here because the confusion matrix gives up
# all the numbers we need.

# P(at risk if positive) = P(Positive if At Risk) X P(At Risk) / P(Positive)
# We can get duped by probabilities that are high only in a specific sample like the vendor's one thousand test patients

""" 3. Receiver Operator Characteristics/Area Under Curve """
# When we are evaluating different ML configurations, we may end up with dozens, hundreads, or thousands of confusion matrices.
# These can be tedious to review, so we can summarize all of them with a receiver operator characteristic (ROC)
# We can also compare different machine learning models by creating separate ROC curves for each
# The "area under the curve (AUC)"" is a good metric for choosing which model to use

# To use the AUC as ascoring metric, change the scoring parameter in the scikit-learn API to use 
# "roc_auc" for a cross-validation

# 12.3: Using the AUC as the scikit-learn parameter

# put Scikit-learn model here
results = cross_val_score(model, X, Y, cv=kfold, scoring='roc_auc')
print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))

""" 4. Class Imbalance """
# Class Imbalancec is when data is not equally represented across every outcome class
# It's a problem in machine learning!
# Class Imbalance is still an open problem with no great solution

# You can collect more data or try different models as well as use confusion matrices and ROC/AUC curves.
# Common Technique is to duplicate samples in the minority class untill it is equally represented in the dataset.

# 12.4: Using the stratify opinion in scikit-learn to balance classes in the data
X, Y = ...
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, stratify=Y)

# SMOTE algorithms for generate synthetic samples of the minority class
# The best ideal to tackle the problem in a way that uses anomaly-detection models

# Conclusion:
# 1. Logistic Regression can predict more than one category rather than just a true/false
# 2. Build separate Logistic Regression modeling whether or not it belongs to that category, and the model that produces the highest probability is the winners
# 3. Statistical modeling we covered the R^2 and p-value
# 4. Machine Learning we explored train_test_splits, confusion matrices, and ROC/AUC
# 5. Try to seek out statics resources more. Its gonna help you someday!

# The End.