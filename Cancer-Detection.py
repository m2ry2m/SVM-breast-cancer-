# Support Vector Machine for Breast Cancer Classification
# -----------------------------------------------

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Load breast cancer dataset
cancer = load_breast_cancer()

# Create a DataFrame with feature data
# cancer['data'] contains the features, and cancer['feature_names'] contains their names
df_feat = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])

# Create a DataFrame with the target values (0 = malignant, 1 = benign)
df_target = pd.DataFrame(cancer['target'], columns=['Cancer'])

# Optional: take a look at the first few rows of data
# print(df_feat.head())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_feat, np.ravel(df_target), test_size=0.3, random_state=101)

# Create a basic Support Vector Classifier model
model = SVC()

# Fit the model with training data
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate model performance before tuning
print("\nInitial Model Performance:")
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# -----------------------------------------------
# Improve model using GridSearchCV
# -----------------------------------------------

# Define a grid of parameters to search
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}

# Create GridSearchCV object with SVC and the parameter grid
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

# Fit GridSearch to the training data
grid.fit(X_train, y_train)

# Display the best parameters found
print("\nBest Parameters:")
print(grid.best_params_)

# Predict again using the best model from GridSearch
grid_predictions = grid.predict(X_test)

# Final evaluation
print("\nFinal Model Performance after GridSearch:")
print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))
