
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load CSV (replace 'your_file.csv' with your actual file name)
df = pd.read_csv('your_file.csv')

# Assume some column names exist; adapt these to your actual dataset
X = df[['feature1', 'feature2', 'feature3']]
y = df['target']

# Split dataset
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
