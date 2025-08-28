import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import torch
from scipy.spatial.transform import Rotation





m = 60
X = np.zeros((m, 3))  # initialize 3D dataset
angles = (np.random.rand(m) ** 3 + 0.5) * 2 * np.pi  # uneven distribution
X[:, 0], X[:, 1] = np.cos(angles), np.sin(angles) * 0.5  # oval
X += 0.28 * np.random.randn(m, 3)  # add more noise
X = Rotation.from_rotvec([np.pi / 29, -np.pi / 20, np.pi / 4]).apply(X)
X += [0.2, 0, 0.2]  # shift a bit

plt.scatter(X[:, 0], X[:, 1])
plt.show()



from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
plt.scatter(X2D[:, 0], X2D[:, 1])
plt.show()

pca = PCA(0.8)
Xp = pca.fit_transform(X)
Xr = pca.inverse_transform(Xp)
print(pca.explained_variance_ratio_)


from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

X_swiss, t = make_swiss_roll(n_samples=1000, noise=0.2)
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_unrolled = lle.fit_transform(X_swiss)


from sklearn.manifold import MDS
mds = MDS(n_components=2, normalized_stress=False)
Xr = mds.fit_transform(X_swiss)



from sklearn.manifold import Isomap
isomap = Isomap(n_components=2)
Xi = isomap.fit_transform(X_swiss)



from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, init="random", learning_rate="auto")
Xt = tsne.fit_transform(X_swiss)



from sklearn.decomposition import KernelPCA
rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
Xr = rbf_pca.fit_transform(X_swiss)

