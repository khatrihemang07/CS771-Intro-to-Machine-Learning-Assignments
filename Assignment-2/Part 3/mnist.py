import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = np.load('mnist_small.pkl', encoding='bytes', allow_pickle=True)
# Initialize t-SNE with 2 output dimensions
X = data['X']
Y = data['Y']

# Ensure Y is one-dimensional
Y = np.squeeze(Y)
# Apply PCA to project the data into 2D
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)

# Visualize PCA results
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
for i in range(10):
    index = (Y == i)
    plt.scatter(pca_result[index, 0], pca_result[index, 1], label=str(i))
plt.title('PCA Visualization')
plt.legend()


# Apply t-SNE to project the data into 2D
tsne = TSNE(n_components=2, random_state=25)
tsne_result = tsne.fit_transform(X)

# Visualize t-SNE results
plt.subplot(1, 2, 2)
for i in range(10):
    index = (Y == i)
    plt.scatter(tsne_result[index, 0], tsne_result[index, 1], label=str(i))
plt.title('t-SNE Visualization')
plt.legend()

plt.show()