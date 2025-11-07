import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

from sklearn.datasets import load_iris
from torch.utils.data import Dataset, DataLoader


class TorchClassifier(nn.Module):

    def __init__(self, fanin, hidden, fanout):
        super().__init__()
        self.layerA = nn.Linear(fanin, hidden)
        self.layerB = nn.Linear(hidden, fanout)
        self.output = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.layerA(x)
        x = torch.sigmoid(x)
        x = self.layerB(x)
        x = self.output(x)
        return x

    def fit(self, X, y, num_epochs=20000, batch_size=256, learning_rate=1e-2):

        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss()

        losses = [0] * num_epochs

        for epoch in range(num_epochs):
            idx = torch.randint(low=0, high=y.size(0), size=(batch_size,))
            Xb = X[idx]
            yb = y[idx].reshape(-1)

            yhat = self.forward(Xb)
            loss = loss_function(yhat, yb)
            loss.backward()
            losses[epoch] = loss.item()
            optimizer.step()
            optimizer.zero_grad()

        plt.plot(losses)


iris = load_iris()
Xiris = iris.data
yiris = iris.target

num_features = Xiris.shape[1]
num_hidden = 6
num_types = len(np.unique(yiris))
print(f"Number of features:{num_features}, Number of Classes:{num_types}")

Xtrain, Xtest, ytrain, ytest = train_test_split(Xiris, yiris, test_size=0.1)


Xtrain = torch.tensor(Xtrain, dtype=torch.float32).reshape(-1, num_features)
ytrain = torch.tensor(ytrain, dtype=torch.long).reshape(-1, 1)
model = TorchClassifier(num_features, num_hidden, num_types)
model.fit(Xtrain, ytrain)


Xt = torch.tensor(Xtest, dtype=torch.float32).reshape(-1, num_features)
yt = torch.tensor(ytest, dtype=torch.long).reshape(-1)
yth = model(Xt)
testloss = nn.CrossEntropyLoss()
testloss(yth.data, yt)
