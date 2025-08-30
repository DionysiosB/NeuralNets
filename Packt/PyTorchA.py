import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

plt.scatter(cars['wt'], cars['mpg'])
X = cars['wt'].values
y = cars['mpg'].values
X = [[data] for data in X]

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
print(model.coef_, model.intercept_)



X = cars['wt'].values
y = cars['mpg'].values
X = torch.tensor(X, dtype=torch.float64).reshape(-1, 1)
y = torch.tensor(y, dtype=torch.float64).reshape(-1, 1)

num_epochs = 100
learning_rate = 1e-2
losses = [0] * num_epochs
w = torch.rand(1, requires_grad=True, dtype=torch.float64)
b = torch.rand(1, requires_grad=True, dtype=torch.float64)

for epoch in range(num_epochs):
    total_loss = 0;
    for p in range(len(y)):
        yhat = w * X[p] + b
        loss = torch.pow((yhat - y[p]), 2)
        loss.backward()

        with torch.no_grad():
            w -= w.grad * learning_rate
            b -= b.grad * learning_rate
            w.grad.zero_()
            b.grad.zero_()

        total_loss += loss.data[0]

    losses[epoch] = total_loss
plt.plot(range(num_epochs), losses)
print(w.item(), b.item())



X = cars['wt'].values
y = cars['mpg'].values

X = torch.tensor(X, dtype = torch.float64).reshape(-1, 1)
y = torch.tensor(y, dtype = torch.float64).reshape(-1, 1)

num_epochs = 3000
learning_rate = 1e-2
lossvector = [0] * num_epochs
w = torch.rand(1, requires_grad=True, dtype=torch.float64)
b = torch.rand(1, requires_grad=True, dtype=torch.float64)

for epoch in range(num_epochs):
    yhat = X * w + b
    loss = torch.mean(torch.pow((yhat - y), 2))
    loss.backward()

    with torch.no_grad():
        w -= w.grad * learning_rate
        b -= b.grad * learning_rate
        w.grad.zero_()
        b.grad.zero_()

    lossvector[epoch] = loss.item()


plt.plot(range(5, num_epochs), lossvector[5:])
print(lossvector[-5:-1])
print(w.item(), b.item()
