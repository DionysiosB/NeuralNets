import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import torch


from sklearn.preprocessing import add_dummy_feature

m = 1000
X = 2 + np.random.randn(m)
y = 3 + 5 * X + np.random.randn(m)
plt.plot(X, y, "b.")
Xb = add_dummy_feature(X.reshape(-1, 1))
y = y.reshape(-1, 1)
beta = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y
print(beta)


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X.reshape(-1, 1), y)
print(model.intercept_ , model.coef_)


num_epochs = 10000
learning_rate = 1e-5
beta = np.random.randn(2, 1)
for epoch in range(num_epochs):
    gradient = Xb.T @ (Xb @ beta - y)
    beta = beta - learning_rate * gradient
print(beta)


num_epochs = 1000
learning_rate = 1e-2
beta = np.random.randn(2, 1)
for epoch in range(num_epochs):
    idx = np.random.choice(m, 1)
    xx = Xb[idx]
    yy = y[idx]
    gradient = xx.T @ (xx @ beta - yy)
    beta = beta - learning_rate * gradient
print(beta)





from sklearn.linear_model import SGDRegressor

model = SGDRegressor()
model.fit(X.reshape(-1, 1), y.reshape(-1))
print(model.intercept_, model.coef_)


m = 10000
X = 2 + 3 * np.random.randn(m)
y = 5 + 3 * X + 2 * X**2 + np.random.randn(m) 
plt.plot(X, y, "b.")

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
Xb = poly.fit_transform(X.reshape(-1, 1))
y = y.reshape(-1, 1)
model = LinearRegression()
model.fit(Xb, y)
print(model.intercept_, model.coef_)





from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
m = 100
X = np.linspace(-5, 5, m).reshape(m, 1)
y = 5 + 7 * X + np.random.randn(m, 1)
plt.plot(X, y, "b.")
model = Ridge(alpha=10000, solver="cholesky")
model.fit(X, y)
model = SGDRegressor(penalty="l2", alpha=100, tol=None, max_iter=1000, eta0=0.01, random_state=42)
model.fit(X, y.ravel())  # y.ravel() because fit() expects 1D targets
print(model.intercept_, model.coef_)



from sklearn.linear_model import Lasso
m = 100
X = np.linspace(-5, 5, m).reshape(m, 1)
y = 5 + 7 * X + np.random.randn(m, 1)
plt.plot(X, y, "b.")
model = Lasso(alpha=50)
model.fit(X, y)
print(model.intercept_, model.coef_)

      

from sklearn.linear_model import ElasticNet
m = 100
X = np.linspace(-5, 5, m).reshape(m, 1)
y = 5 + 7 * X + np.random.randn(m, 1)
plt.plot(X, y, "b.")
model = Lasso(alpha=50)
model.fit(X, y)
print(model.intercept_, model.coef_)






from sklearn.linear_model import ElasticNet
m = 100
X = np.linspace(-5, 5, m).reshape(m, 1)
y = 5 + 7 * X + np.random.randn(m, 1)
plt.plot(X, y, "b.")
model = ElasticNet(alpha=10, l1_ratio=0.5)
model.fit(X, y)
print(model.intercept_, model.coef_)




from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

X = iris.data[["petal width (cm)"]].values
y = iris.target_names[iris.target] == 'virginica'
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

print(model.intercept_, model.coef_)

xb = np.linspace(0, 3, 1000).reshape(-1, 1)
yb = model.predict_proba(xb)
threshold = 0.9
plt.plot(xb, yb[:, 1], 'b.', xb, yb[:, 1] > threshold, 'r.')
plt.show()
