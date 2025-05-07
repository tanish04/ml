LIBRARIES:

import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline

from sklearn.naive_bayes import GaussianNB

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.cluster import AgglomerativeClustering, KMeans

from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score



LOAD DATASET:

Load Iris dataset

iris = load_iris()

X = iris.data[:, 0].reshape(-1, 1) # Sepal length (1st feature)

y = iris.data[:, 2].reshape(-1, 1)


TRAIN-TEST SPLIT:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


SIMPLE LINEAR REGRESSION

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.scatter(X_test, y_test, color='green', label='Actual')

plt.plot(X_test, y_pred, color='red', label='Predicted')

plt.xlabel('Sepal Length (cm)')

plt.ylabel('Petal Length (cm)')

plt.title('Linear Regression on Iris Dataset')

plt.legend()

plt.show()


MULTIPLE LINEAR REGRESSION

iris = load_iris()

X = iris.data[:, [0, 1, 3]] # Sepal Length, Sepal Width, Petal Width

y = iris.data[:, 2]

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.scatter(y_test, y_pred, color='purple')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')

plt.xlabel('Actual Petal Length')

plt.ylabel('Predicted Petal Length')

plt.title('Actual vs Predicted Petal Length')

plt.grid(True)

plt.show()


POLYNOMIAL REGRESSION

degree = 3

model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.scatter(X, y, color='green', label='Actual Data')

plt.plot(X_plot, y_plot, color='red', linewidth=2, label=f'Degree {degree} Fit')

plt.xlabel('Sepal Length (cm)')

plt.ylabel('Petal Length (cm)')

plt.title(f'Polynomial Regression (Degree {degree}) on Iris')

plt.legend()

plt.grid(True)

plt.show()


RIDGE REGRESSION

model = Ridge(alpha=0.1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


EVALUATE

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)


NAÏVE BAYES

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


DECISION TREE

model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


KNN

k = 5

model = KNeighborsClassifier(n_neighbors=k)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


LOGISTIC REGRESSION

model = LogisticRegression(max_iter=200, multi_class='multinomial', solver='lbfgs')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


SVM

model = SVC(kernel='rbf', C=1.0, gamma='scale') # Try 'linear', 'poly', or 'rbf'

model.fit(X_train, y_train)

y_pred = model.predict(X_test)




EVALUATE

accuracy = accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred, target_names=target_names))

cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix

plt.figure(figsize=(8, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names) plt.xlabel("Predicted Labels")

plt.ylabel("True Labels")

plt.title("Confusion Matrix")

plt.show()


K MEANS CLUSTERING

kmeans = KMeans(n_clusters=3, random_state=42)

y_kmeans = kmeans.fit_predict(X)

# Evaluate the clustering performance using silhouette score

sil_score = silhouette_score(X, y_kmeans)

print(f"Silhouette Score: {sil_score:.2f}")

# Visualize the clusters using PCA (to reduce to 2D for plotting)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

# Plotting the clusters

plt.figure(figsize=(8, 6))

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', marker='o', edgecolor='k')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label="Centroids")

plt.title('K-Means Clustering (Iris Dataset) with PCA Projection')

plt.xlabel('PCA 1')

plt.ylabel('PCA 2')

plt.legend()

plt.show()


HIERARCHICAL

# Perform Hierarchical Clustering (Agglomerative)

agg_clust = AgglomerativeClustering(n_clusters=3, linkage='ward')

y_agg_clust = agg_clust.fit_predict(X)

# Plot the Dendrogram

linked = linkage(X, 'ward') # 'ward' minimizes variance within clusters

plt.figure(figsize=(10, 7))

dendrogram(linked, labels=iris.target_names[y], orientation='top', distance_sort='descending', show_leaf_counts=True)

plt.title('Hierarchical Clustering Dendrogram')

plt.xlabel('Sample Index or (Cluster Size)')

plt.ylabel('Distance')

plt.show()

# Evaluation
print("Linear Regression R^2:", r2_score(y_test, y_pred_lr))
print("Ridge Regression R^2:", r2_score(y_test, y_pred_ridge))
print("Lasso Regression R^2:", r2_score(y_test, y_pred_lasso))
