import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


class TorchRegressor(nn.Module):

    def __init__(self, fanin, fanout):
        super(TorchRegressor, self).__init__()
        self.layer = nn.Linear(fanin, fanout)

    def forward(self, x):
        return self.layer(x)

    def fit(self, X, y, num_epochs=1000, learning_rate=1e-3):

        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        loss_hist = [0] * num_epochs

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            yhat = self.forward(X)
            loss = criterion(yhat, y)
            loss_hist[epoch] = loss.item()
            loss.backward()
            optimizer.step()

        plt.plot(loss_hist)
        plt.show()
        print(f"Final Loss:{loss_hist[-1]}")
        print(list(self.parameters()))


numpoints = 1000
xx = np.linspace(-5, 5, numpoints)
yy = 3 * xx + 7 + np.random.randn(numpoints)
# xx, yy = make_regression(n_samples=numpoints, n_features=1, noise=20)
X = torch.tensor(xx, dtype=torch.float).view(-1, 1)
y = torch.tensor(yy, dtype=torch.float).view(-1, 1)

model = TorchRegressor(1, 1)
model.fit(X, y, num_epochs=1000, learning_rate=1e-2)
print(f"Estimated Parameters: Slope:{model.layer.weight} Bias:{model.layer.bias}") 
yhat = model.forward(X).detach().numpy()
plt.scatter(xx, yy)
plt.scatter(xx, yhat)
plt.show()
