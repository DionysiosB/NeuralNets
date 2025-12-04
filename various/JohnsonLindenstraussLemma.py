import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

num_vec = 10000
num_dim = 100
num_steps = 250

X = torch.randn(num_dim, num_vec)
X /= X.norm(p=2, dim=0, keepdim=True)
X.requires_grad_(True)
optimizer = torch.optim.Adam([X], lr=0.01)
losses = [0] * num_steps

dot_diff_cutoff = 0.01
big_id = torch.eye(num_vec, num_vec)

for step_num in tqdm(range(num_steps)):
    optimizer.zero_grad()
    dot_products = X.T @ X
    # Punish deviation from orthogonal
    diff = dot_products - big_id
    loss = (diff.abs() - dot_diff_cutoff).relu().sum() + num_vec * diff.diag().pow(2).sum()
    loss.backward()
    optimizer.step()
    losses[step_num] = loss.item()

plt.plot(losses)
plt.grid(1)
plt.show()

dot_products = X.T @ X
norms = torch.sqrt(torch.diag(dot_products))
normed_dot_products = dot_products / torch.outer(norms, norms)
angles_degrees = torch.rad2deg(torch.acos(normed_dot_products.detach()))
# Use this to ignore self-orthogonality.
self_orthogonality_mask = ~(torch.eye(num_vec, num_vec).bool())
plt.hist(angles_degrees[self_orthogonality_mask].numpy().ravel(), bins=1000, range=(80, 100))
plt.grid(1)
plt.show()
