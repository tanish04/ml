
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Convert to DataFrame
df = pd.DataFrame(data=np.hstack((X, y)), columns=['X', 'y'])

# Add a column of ones to X for the intercept term
X_b = np.c_[np.ones((len(X), 1)), X]  # add x0 = 1 to each instance

# Calculate the best theta using the Normal Equation: (X'X)^(-1)X'y
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Predictions
y_pred = X_b.dot(theta_best)

# Plotting
plt.figure(figsize=(8, 6))
sns.scatterplot(x='X', y='y', data=df, label='Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression without sklearn')
plt.legend()
plt.grid(True)
plt.show()

# Evaluation Metrics
residuals = y - y_pred
rss = np.sum(residuals ** 2)
mse = rss / len(y)
rse = np.sqrt(mse)

tss = np.sum((y - np.mean(y)) ** 2)
r2_score = 1 - (rss / tss)

print(f"Intercept: {theta_best[0][0]:.4f}")
print(f"Slope: {theta_best[1][0]:.4f}")
print(f"RSS: {rss:.4f}")
print(f"RSE: {rse:.4f}")
print(f"R-squared: {r2_score:.4f}")
