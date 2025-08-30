import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
X = cars['wt'].values
y = cars['mpg'].values

X = torch.tensor(X, dtype = torch.float32).reshape(-1, 1)
y = torch.tensor(y, dtype = torch.float32).reshape(-1, 1)

class LinearRegressionTorch(nn.Module):

    def __init__(self, fanin, fanout):
        super().__init__()
        self.layer = nn.Linear(fanin, fanout, dtype=torch.float32)

    def forward(self, data):
        return self.layer(data)


    def fit(self, X, ytrue, num_epochs = 1, learning_rate = 0.01):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr = learning_rate)
        loss_function = nn.MSELoss()

        losses = [0] * num_epochs

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            yhat = self.layer(X)
            loss = loss_function(yhat, ytrue)
            loss.backward()
            optimizer.step()
            losses[epoch] = loss.item()

        print(f"Final Loss:{loss.item()}")
        plt.plot(losses[100:])
        plt.show()



model = LinearRegressionTorch(1, 1)
model.fit(X, y, num_epochs=1000, learning_rate=0.02)
print(f"Weight:{model.layer.weight.item()} Bias:{model.layer.bias.item()}")
