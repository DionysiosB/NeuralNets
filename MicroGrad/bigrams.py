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
mat = np.zeros((B + 1, B + 1))
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

for _ in range(N):
    asciilist = list()
    while True:
        prev = asciilist[-1] if len(asciilist) else B
        next = np.random.choice(range(B + 1), p=mat[prev])
        if next == B: break
        asciilist.append(next)
    output = ''.join(map(chr, [int(ord('a') + x) for x in asciilist]))
    print(output
