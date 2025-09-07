import numpy as np


def kmeans(X, k, max_iters=100):
  """
  Perform K-Means clustering on the input data X with k clusters.

  Parameters:
      X (numpy.ndarray): Input data of shape (n_samples, n_features).
      k (int): Number of clusters.
      max_iters (int): Maximum number of iterations to run the algorithm.

  Returns:
      centroids (numpy.ndarray): Final cluster centroids of shape (k, n_features).
      labels (numpy.ndarray): Cluster assignments for each sample in X.
  """

  # Step 1: Initialize centroids randomly
  n_samples, n_features = X.shape
  centroids = X[np.random.choice(n_samples, k, replace=False)]

  for _ in range(max_iters):
    # Step 2: Assign each point to the nearest centroid
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)

    # Step 3: Update centroids based on the mean of assigned points
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

    # Check for convergence (centroids do not change)
    if np.all(centroids == new_centroids):
      break

    centroids = new_centroids

  return centroids, labels


# Example usage
if __name__ == '__main__':
  # Generate some random data
  X = np.random.rand(100, 10)  # 100 samples with 2 features each
  k = 5  # Number of clusters

  # Run K-Means
  centroids, labels = kmeans(X, k)

  print('Centroids:\n', centroids)
  print('Labels:\n', labels)
