import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import torch


from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target




from sklearn.mixture import GaussianMixture
mix = GaussianMixture(n_components=3, n_init=10)
ypred = mix.fit_predict(X)


from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

blob_centers = np.array([[ 0.2,  2.3], [-1.5 ,  2.3], [-2.8,  1.8], [-2.8,  2.8], [-2.8,  1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=7)

kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
y_pred = kmeans.fit_predict(X)





from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.05)
dbscan = DBSCAN(eps=0.05, min_samples=5)
ypred = dbscan.fit_predict(X)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.05)
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X, y)
ypred = knn.predict(X)




from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=2, gamma=100)
model.fit(X) ## No predict


from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=2)
model.fit(X) ## No predict  


from sklearn.mixture import GaussianMixture
model = GaussianMixture(n_components=2)
model.fit(X) ## No predict
#print(model.weights_, model.means_, model.covariances_)
ypred = model.predict(X)
