import numpy as np
import torch
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Tuple



block_size = 3          # Number of characters used to predict the next one
B  = 26                 # Size of the alphabet
ND = 6                  # Number of embedding dimensions
NH = 200                # Number of nodes in the hidden layer
BATCH_SIZE     = 1000   # How many samples for each batch
NUM_ITERATIONS = 5000   # How many epochs to run backprop for
learning_rate  = 0.02   # Learning rate
fraction_train = 0.8    # Fraction of the dataset to be used for training
fraction_valid = 0.1    # Fraction of the dataset to be used for validation
window_size    = 100    # Window size to plot smoothed list of loss history
random.seed(0)
g  = torch.Generator().manual_seed(0)



def getnum(x : str, default : int) -> int:
    return (ord(x) - ord('a')) if x.isalpha() else default

def getchar(x : int, default : str) -> str:
    y = chr(ord('a') + x)
    return y if y.isalpha() else default

def build_dataset(words : List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    X, Y = list(), list()
    for w in words:
        if len(w) < block_size: continue
        w = "." * block_size + w + '.'
        for p in range(block_size, len(w)):
            prev = w[(p - block_size) : p]
            prev = [getnum(x, B) for x in prev]
            next = getnum(w[p], B)
            X.append(prev)
            Y.append(next)
    return torch.tensor(X), torch.tensor(Y)


with open("input.txt", "r") as f: words = f.read().splitlines()
random.shuffle(words)

n1 = int(fraction_train * len(words))
n2 = int((fraction_train + fraction_valid) * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xvd, Yvd = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

print(f"Training set shape: {Xtr.shape}")
print(f"Validation set shape: {Xtr.shape}")
print(f"Test set shape: {Xtr.shape}")






C  = torch.randn((B + 1, ND), generator=g)
Wh = torch.randn((block_size * ND, NH), generator=g)
bh = torch.randn(NH, generator=g)
Wt = torch.randn((NH, B + 1), generator=g)
bt = torch.randn((B + 1), generator=g)

all_params =[C, Wh, bh, Wt, bt]
for param in all_params: param.requires_grad = True

num_params = sum([param.nelement() for param in all_params])
print(f"Total number of parameters: {num_params}")



loss_hist = [0] * NUM_ITERATIONS

for iter in range(NUM_ITERATIONS):
    indices = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,))
    embeddings = C[Xtr[indices]]
    hidden = torch.tanh(torch.add(torch.matmul(embeddings.view(-1, block_size * ND) , Wh) , bh))
    logits = torch.add(torch.matmul(hidden, Wt), bt)
    loss = F.cross_entropy(logits, Ytr[indices])

    for param in all_params: param.grad = None
    loss.backward()
    for param in all_params: param.data -= learning_rate * param.grad
    loss_hist[iter] = torch.log(loss).item()

smooth_losses = smoothed_data = [
    np.mean(loss_hist[max(0, p - window_size) : min(len(loss_hist), p + window_size + 1)])
    for p in range(len(loss_hist))
]

plt.figure(figsize=(20, 7))
plt.plot(range(len(loss_hist)), loss_hist)
plt.plot(range(len(smooth_losses)), smooth_losses);
plt.show()

vd_embeddings = C[Xvd]
vd_hidden = torch.tanh(vd_embeddings.view(-1, block_size * ND) @ Wh + bh)
vd_logits = vd_hidden @ Wt + bt
vd_loss = F.cross_entropy(vd_logits, Yvd)

te_embeddings = C[Xte]
te_hidden = torch.tanh(te_embeddings.view(-1, block_size * ND) @ Wh + bh)
te_logits = te_hidden @ Wt + bt
te_loss = F.cross_entropy(te_logits, Yte)

print(f"TrainLoss:\t{smooth_losses[-1]}\nValidationLoss:\t{vd_loss}\nTestLoss:\t{te_loss}")

plt.figure(figsize=(8,8))
plt.scatter(C[:, 0].data, C[:,1].data, s=300)
for p in range(C.shape[0]):
    plt.text(C[p, 0].item(), C[p, 1].item(), getchar(p, '.'), ha='center', va='center', color='white')
plt.grid('minor')
plt.show()



num_examples = 20
print(f"Here are {num_examples} examples:")
for _ in range(num_examples):
    rw = ""
    context = [B] * block_size
    while True:
      embeddings = C[torch.tensor([context])]
      hidden = torch.tanh(embeddings.view(1, -1) @ Wh + bh)
      logits = hidden @ Wt + bt
      probs = F.softmax(logits, dim=1)
      idx = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [idx]
      if idx == B: break
      rw += getchar(idx, '.')
    print(rw
