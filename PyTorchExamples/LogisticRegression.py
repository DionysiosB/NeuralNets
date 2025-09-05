import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class TorchLogisticRegression(nn.Module):

    def __init__(self, fanin, fanout):
        super(TorchLogisticRegression, self).__init__()
        self.layer = nn.Sequential(nn.Linear(fanin, fanout), nn.Sigmoid())

    def forward(self, x):
        return self.layer(x)

    def fit(self, X, y, num_epochs=1000, learning_rate=0.01):

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()

        loss_hist = [0] * num_epochs

        for epoch in range(num_epochs):

            yhat = self.forward(X)
            loss = criterion(yhat, y)
            loss_hist[epoch] = loss.item()
            if not (epoch % 10): print(f"Epoch:{epoch} Loss:{loss.item()}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad

        plt.plot(loss_hist)
        plt.show()
        for param in list(model.parameters()): print(param)

        
bc = load_breast_cancer()
Xbc = bc.data
ybc = bc.target

Xtrain, Xtest, ytrain, ytest = train_test_split(Xbc, ybc, test_size=0.2)
scaler = StandardScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)


num_features = Xtrain.shape[1]
X = torch.tensor(Xtrain, dtype=torch.float)
y = torch.tensor(ytrain, dtype=torch.float).view(-1, 1)

model = TorchLogisticRegression(num_features, 1)
model.fit(X, y, num_epochs=500, learning_rate=0.003)

with torch.no_grad():
    ypred = model(torch.tensor(Xtest, dtype=torch.float))
    ytest = torch.tensor(ytest, dtype=torch.float).view(-1, 1)
    loss_fcn = nn.BCELoss()
    error = loss_fcn(ypred, ytest)
    print(f"Test Error:{error}")

    ypred_binary = np.round(ypred.numpy())
    score = np.sum(ypred_binary == ytest.numpy()) / len(ypred_binary)
    print(f"Final Score:{score}")


