import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import torch


from sklearn.svm import SVC
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

rows = (iris.target == 0) | (iris.target == 1)

Xa = iris.data[["petal length (cm)", "petal width (cm)"]].values
Xa = Xa[rows]
ya = y[rows]

model = SVC(kernel="linear", C=1000)
model.fit(X, y)

plt.scatter(Xa[:, 0], Xa[:, 1], c=ya)
plt.show()


from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = make_moons(n_samples=500, noise=0.1)
clf = make_pipeline(
    PolynomialFeatures(degree=3),
    StandardScaler(),
    SVC(kernel="linear", C=1000)
)

clf.fit(X, y)


from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = make_moons(n_samples=500, noise=0.1)
clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="poly", degree=3,C=1000)
)
clf.fit(X, y)


from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = make_moons(n_samples=500, noise=0.1)
clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", degree=3,C=1000)
)
clf.fit(X, y)




from sklearn.svm import LinearSVR
from sklearn.svm import SVR

X = 2 * np.random.randn(50)
y = 4 + 3 * X + np.random.randn(50)
model = LinearSVR(epsilon=0.5)
model.fit(X.reshape(-1, 1), y)
print(model.intercept_, model.coef_)

X = 2 * np.random.randn(50)
y = 5 + 3 * X + 2*X**2 + np.random.randn(50)
model = SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1)
model.fit(X.reshape(-1, 1), y)
print(model.support_vectors_)

