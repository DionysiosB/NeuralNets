import math
import random

###############################################################
# Utility functions (NO matrix multiplication, for-loops only)
###############################################################

def matmul(A, B):
    """
    A: [M, K]
    B: [K, N]
    returns: [M, N]
    """
    M = len(A)          # scalar
    K = len(A[0])       # scalar
    N = len(B[0])       # scalar
    out = [[0.0 for _ in range(N)] for _ in range(M)]   # [M, N]

    for i in range(M):                 # i: 0..M-1
        for j in range(N):             # j: 0..N-1
            s = 0.0                    # scalar accum
            for k in range(K):         # k: 0..K-1
                s += A[i][k] * B[k][j]
            out[i][j] = s
    return out


def layernorm(x, gamma, beta, eps=1e-5):
    """
    x:     [T, D]
    gamma: [D]
    beta:  [D]
    """
    T = len(x)          # scalar
    D = len(x[0])       # scalar
    out = [[0.0]*D for _ in range(T)]   # [T, D]

    for t in range(T):                 # t: 0..T-1
        mean = sum(x[t]) / D          # scalar
        var = sum((x[t][i] - mean) ** 2 for i in range(D)) / D   # scalar
        inv_std = 1.0 / math.sqrt(var + eps)   # scalar

        for i in range(D):            # i: 0..D-1
            out[t][i] = gamma[i] * (x[t][i] - mean) * inv_std + beta[i]
    return out


###############################################################
# Self-attention (single head)
###############################################################

def self_attention_one_head(x, Wq, Wk, Wv, Wo):
    """
    x:  [T, D]
    Wq: [D, D]
    Wk: [D, D]
    Wv: [D, D]
    Wo: [D, D]
    """
    T = len(x)          # scalar
    D = len(x[0])       # scalar

    Q = matmul(x, Wq)   # [T, D]
    K = matmul(x, Wk)   # [T, D]
    V = matmul(x, Wv)   # [T, D]

    scores = [[0.0 for _ in range(T)] for _ in range(T)]    # [T, T]

    for t1 in range(T):                                 # t1: 0..T-1
        for t2 in range(T):                             # t2: 0..T-1
            if t2 > t1:
                scores[t1][t2] = -1e9                   # masked
            else:
                s = 0.0                                 # scalar accum
                for d in range(D):                     # d: 0..D-1
                    s += Q[t1][d] * K[t2][d]
                scores[t1][t2] = s / math.sqrt(D)

    att = [[0.0 for _ in range(T)] for _ in range(T)]      # [T, T]

    for t1 in range(T):
        row = scores[t1]                                # [T]
        m = max(row)                                    # scalar
        exps = [math.exp(v - m) for v in row]           # [T]
        Z = sum(exps)                                   # scalar
        for t2 in range(T):
            att[t1][t2] = exps[t2] / Z                  # scalar

    out = [[0.0 for _ in range(D)] for _ in range(T)]     # [T, D]

    for t1 in range(T):
        for t2 in range(T):
            for d in range(D):
                out[t1][d] += att[t1][t2] * V[t2][d]

    out = matmul(out, Wo)   # [T, D]
    return out


###############################################################
# Feed Forward Network
###############################################################

def feed_forward(x, W1, b1, W2, b2):
    """
    x:  [T, D]
    W1: [D, M]
    b1: [M]
    W2: [M, D]
    b2: [D]
    """
    T = len(x)              # scalar
    D = len(x[0])           # scalar
    M = len(b1)             # scalar

    h = matmul(x, W1)       # [T, M]
    for t in range(T):
        for m in range(M):
            h[t][m] += b1[m]

    for t in range(T):
        for m in range(M):
            if h[t][m] < 0.0:
                h[t][m] = 0.0

    o = matmul(h, W2)       # [T, D]
    for t in range(T):
        for d in range(D):
            o[t][d] += b2[d]

    return o


###############################################################
# Transformer Block
###############################################################

def transformer_block(x, p):
    """
    x: [T, D]
    p: parameter dictionary
    """
    T = len(x)          # scalar
    D = len(x[0])       # scalar

    ln1 = layernorm(x, p["ln1_gamma"], p["ln1_beta"])    # [T, D]
    sa  = self_attention_one_head(ln1, p["Wq"], p["Wk"], p["Wv"], p["Wo"])  # [T, D]

    res1 = [[x[t][d] + sa[t][d] for d in range(D)] for t in range(T)]   # [T, D]

    ln2 = layernorm(res1, p["ln2_gamma"], p["ln2_beta"])  # [T, D]

    ff = feed_forward(ln2, p["W1"], p["b1"], p["W2"], p["b2"])  # [T, D]

    out = [[res1[t][d] + ff[t][d] for d in range(D)] for t in range(T)] # [T, D]
    return out


###############################################################
# Full Transformer Inference
###############################################################

def next_token_logits(tokens, model):
    """
    tokens: [T]
    returns logits for last token: [V]
    """
    T = len(tokens)        # scalar
    D = model["D"]         # scalar
    V = model["V"]         # scalar

    x = [[0.0 for _ in range(D)] for _ in range(T)]       # [T, D]

    for t in range(T):
        tok = tokens[t]                                   # scalar
        tok_emb = model["token_emb"][tok]                 # [D]
        pos_emb = model["pos_emb"][t]                     # [D]
        for d in range(D):
            x[t][d] = tok_emb[d] + pos_emb[d]             # scalar

    for blk in model["blocks"]:
        x = transformer_block(x, blk)                     # [T, D]

    x = layernorm(x, model["ln_f_gamma"], model["ln_f_beta"])   # [T, D]

    logits = matmul(x, model["lm_head"])   # [T, V]
    return logits[-1]                      # [V]


###############################################################
# Autoregressive generation
###############################################################

def generate(model, prompt_tokens, L):
    """
    prompt_tokens: [T_start]
    Returns: [T_start + L]
    """
    tokens = list(prompt_tokens)    # [T_current]
    V = model["V"]                  # scalar

    for _ in range(L):
        logits = next_token_logits(tokens, model)   # [V]

        m = max(logits)                   # scalar
        exps = [math.exp(v - m) for v in logits]  # [V]
        Z = sum(exps)                    # scalar
        probs = [e / Z for e in exps]    # [V]

        next_tok = max(range(V), key=lambda i: probs[i])   # scalar
        tokens.append(next_tok)        # [T_current+1]

    return tokens


###############################################################
# Random tiny model for testing
###############################################################

def random_matrix(m, n):
    return [[random.uniform(-0.02, 0.02) for _ in range(n)] for _ in range(m)]   # [m, n]

def random_vector(n):
    return [random.uniform(-0.02, 0.02) for _ in range(n)]  # [n]


def build_toy_model(V=200, D=32, M=64, L=2, T_max=128):
    model = {
        "V": V,                          # vocab size
        "D": D,                          # embed dim
        "token_emb": random_matrix(V, D),  # [V, D]
        "pos_emb": random_matrix(T_max, D), # [T_max, D]
        "blocks": [],                     # list[L]
        "ln_f_gamma": random_vector(D),   # [D]
        "ln_f_beta": random_vector(D),    # [D]
        "lm_head": random_matrix(D, V)    # [D, V]
    }

    for _ in range(L):
        blk = {
            "Wq": random_matrix(D, D),  # [D, D]
            "Wk": random_matrix(D, D),  # [D, D]
            "Wv": random_matrix(D, D),  # [D, D]
            "Wo": random_matrix(D, D),  # [D, D]
            "ln1_gamma": random_vector(D),   # [D]
            "ln1_beta": random_vector(D),    # [D]
            "W1": random_matrix(D, M),       # [D, M]
            "b1": random_vector(M),          # [M]
            "W2": random_matrix(M, D),       # [M, D]
            "b2": random_vector(D),          # [D]
            "ln2_gamma": random_vector(D),   # [D]
            "ln2_beta": random_vector(D)     # [D]
        }
        model["blocks"].append(blk)

    return model


###############################################################
# Demo
###############################################################

if __name__ == "__main__":

    random.seed(0)
    model = build_toy_model()
    prompt = [10, 3, 17]           # [3]
    result = generate(model, prompt, L=5)  # produces [8]
    print("Generated:", result)
