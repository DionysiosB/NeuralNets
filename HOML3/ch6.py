import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import torch


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris.target

dtc = DecisionTreeClassifier(max_depth=2, random_state=42)
dtc.fit(X, y)
dtc.predict([[5, 1.5]])
dtc.predict_proba([[5, 1.5]])



from sklearn.datasets import make_moons
X, y = make_moons(n_samples=150, noise=0.2, random_state=42)
#dtc = DecisionTreeClassifier(random_state=42)
dtc = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
dtc.fit(X, y)


from sklearn.tree import DecisionTreeRegressor

m = 200
X = np.random.randn(m)
y = 3 * X * X + np.random.randn(m)

model = DecisionTreeRegressor(max_depth=2)
model.fit(X.reshape(-1, 1), y)




from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris


iris = load_iris()
pipeline = make_pipeline(StandardScaler(), PCA())
X = pipeline.fit_transform(iris.data)
y = iris.target
model = DecisionTreeClassifier()
model.fit(X, y)
