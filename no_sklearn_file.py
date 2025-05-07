//LINEAR REGRESSION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate sample dataset or load your own CSV
# For demo: creating dummy CSV-like data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Feature: e.g., Study Hours
y = 2.5 * X + np.random.randn(100, 1) * 2  # Target: e.g., Exam Scores

# Convert to DataFrame and save to CSV (simulate loading real data)
data = pd.DataFrame({'Hours': X.flatten(), 'Scores': y.flatten()})
data.to_csv('student_scores.csv', index=False)

# Read the CSV data
df = pd.read_csv('student_scores.csv')
X = df['Hours'].values
y = df['Scores'].values

# Mean normalization
X_mean = np.mean(X)
y_mean = np.mean(y)
numerator = np.sum((X - X_mean) * (y - y_mean))
denominator = np.sum((X - X_mean)**2)
slope = numerator / denominator
intercept = y_mean - slope * X_mean

# Prediction
y_pred = slope * X + intercept

# Evaluation (R^2 score)
ss_total = np.sum((y - y_mean)**2)
ss_res = np.sum((y - y_pred)**2)
r2 = 1 - (ss_res / ss_total)

print(f"Slope: {slope:.2f}, Intercept: {intercept:.2f}")
print(f"R-squared Score: {r2:.4f}")

# Plot
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Linear Regression without sklearn")
plt.legend()
plt.grid(True)
plt.show()

//K MEANS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create synthetic data
np.random.seed(0)
cluster_1 = np.random.randn(50, 2) + [5, 5]
cluster_2 = np.random.randn(50, 2) + [15, 15]
cluster_3 = np.random.randn(50, 2) + [25, 5]
data = np.vstack([cluster_1, cluster_2, cluster_3])
df = pd.DataFrame(data, columns=['X', 'Y'])

# K-Means from scratch
def euclidean(a, b):
    return np.sqrt(np.sum((a - b)**2))

def k_means(X, k=3, max_iters=100):
    # Random initial centroids
    np.random.seed(42)
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        for point in X:
            distances = [euclidean(point, centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)
        prev_centroids = centroids.copy()
        centroids = [np.mean(cluster, axis=0) if cluster else centroids[i] for i, cluster in enumerate(clusters)]
        if np.allclose(prev_centroids, centroids):
            break
    labels = np.zeros(len(X))
    for i, point in enumerate(X):
        distances = [euclidean(point, centroid) for centroid in centroids]
        labels[i] = np.argmin(distances)
    return labels, centroids

X = df[['X', 'Y']].values
labels, centroids = k_means(X, k=3)

# Plotting results
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for i in range(3):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i+1}')
plt.scatter(*zip(*centroids), color='black', marker='x', s=200, label='Centroids')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-Means Clustering without sklearn')
plt.legend()
plt.grid(True)
plt.show()

//
