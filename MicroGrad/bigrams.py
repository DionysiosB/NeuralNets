import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


input_file = "input.txt"
sep = '|'
B = 26 # Number of characters
N = 10 # Number of words to output

words = list()
with open(input_file, "r") as f: words = f.read().splitlines()
fs = sep + sep.join(words) + sep


# Construct the 2D transition array
mat = np.ones((B + 1, B + 1))  #np.ones instead of np.zeros to avoid any log infinities
for p in range(1, len(fs)):
    prev = B if fs[p - 1] == sep else (ord(fs[p - 1]) - ord('a'))
    next = B if fs[p] == sep else (ord(fs[p]) - ord('a'))
    mat[prev][next] += 1


# Show the probabilities
plt.imshow(mat)
plt.show()

# Normalize to probabilities
for row in range(B + 1):
    row_count = np.sum(mat[row])
    for col in range(B + 1): mat[row][col] /= row_count


# Calculate lower bound for negative log loss
nll = 0.0
for p in range(1, len(fs)):
    prev = B if fs[p - 1] == sep else (ord(fs[p - 1]) - ord('a'))
    next = B if fs[p] == sep else (ord(fs[p]) - ord('a'))
    nll -= np.log(mat[prev][next])

nll /= (len(fs) - 1)
print(f"\n\nAverage Lower Bound for Negative Log Likelihood:{nll}\n")




# Output N sampls and calculate log loss for the sample
output = [''] * N
nll = 0.0
cnt = 0

for p in range(N):
    asciilist = list()
    while True:
        prev = asciilist[-1] if len(asciilist) else B
        next = np.random.choice(range(B + 1), p=mat[prev])
        nll -= np.log(mat[prev][next])
        cnt += 1
        if next == B: break
        asciilist.append(next)
    output[p] = ''.join(map(chr, [int(ord('a') + x) for x in asciilist]))


#Results
nll /= cnt
print(f"\n\nNegative Log Likelihood:{nll}\n")

print("\nPredicted Words:\n")
for x in output: print(x) 




#### NN model ####

import torch
import torch.nn.functional as F

num_epochs = 10

# Construct the input/output matrices
sv = [B if c == sep else (ord(c) - ord('a')) for c in fs]
xc = F.one_hot(torch.tensor(sv[:-1]), num_classes=B + 1).float()
ys = torch.tensor(sv[1:])

W = torch.randn((B + 1, B + 1), requires_grad=True)

for epoch in range(num_epochs):
    logits = xc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(len(xc)), ys].log().mean() + 0.1 * torch.mean(W**2)
    print(f"Epoch:{epoch}, Loss:{loss.item()}")
    loss.backward()
    W.data += -50.0 * W.grad
    W.grad = None



## Create some examples

output = [''] * N
for p in range(N):
    asciilist = list()
    while True:
        prev = asciilist[-1] if len(asciilist) else B
        probs = W[prev, :].exp()
        probs = probs / probs.sum()
        next = torch.multinomial(probs, num_samples=1, replacement=True).item()
        nll -= np.log(mat[prev][next])
        cnt += 1
        if next == B: break
        asciilist.append(next)
    output[p] = ''.join(map(chr, [int(ord('a') + x) for x in asciilist]))


print(f"\n\nAverage NLL:{nll/cnt}\n\n")
for x in output: print(x)

