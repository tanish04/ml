
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load Iris dataset
iris = load_iris()
X = iris.data[:, [0, 1, 3]]  # Sepal Length, Sepal Width, Petal Width
y = iris.data[:, 2]          # Petal Length

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Ridge Regression
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

# Evaluation
print("Linear Regression R^2:", r2_score(y_test, y_pred_lr))
print("Ridge Regression R^2:", r2_score(y_test, y_pred_ridge))
print("Lasso Regression R^2:", r2_score(y_test, y_pred_lasso))
