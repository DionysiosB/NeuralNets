import math
import random

def matmul(A, B):
    # A : [M x K]
    # B : [K x N]
    # R : [M x N]
    
    nr = len(A)          # scalar
    nc = len(B[0])       # scalar

    R = [[0.0 for _ in range(nc)] for _ in range(nr)]   # [M x N]

    nk = len(A[0])       # scalar
    assert len(B) == nk , "Number of columns in the first matrix should equal number of rows in the second"

    for row in range(nr):
        for col in range(nc):
            for u in range(nk): R[row][col] += A[row][u] * B[u][col]
    return R


def layernorm(x, gamma, beta, eps=1e-5):
    
    # x:     [T x D]
    # gamma: [1 x D]
    # beta:  [1 x D]
    
    T = len(x)          # scalar
    D = len(x[0])       # scalar
    y = [[0.0 for _ in range(D)] for _ in range(T)]   # [T x D]

    for t in range(T):
        mean  = sum(x[t]) / D
        stdev = math.sqrt(sum([u ** 2 for u in x[t]]) / D - mean ** 2)
        for d in range(D): y[t][d] = gamma[d] * (x[t][d] - mean) / (stdev + eps) + beta[d]

    return y


def attention_head(x, Wq, Wk, Wv, Wo):
    
    # x:  [T x D]
    # Wq: [D x D]
    # Wk: [D x D]
    # Wv: [D x D]
    # Wo: [D x D]
    
    T = len(x)          # scalar
    D = len(x[0])       # scalar

    Q = matmul(x, Wq)   # [T x D]
    K = matmul(x, Wk)   # [T x D]
    V = matmul(x, Wv)   # [T x D]

    scores = [[0.0 for _ in range(T)] for _ in range(T)]    # [T x T]

    for row in range(T):
        for col in range(T):
            if row < col: continue    # masked
            s = 0.0
            for d in range(D): s += Q[row][d] * K[col][d]
            scores[row][col] = s / math.sqrt(D)

    attention = [[0.0 for _ in range(T)] for _ in range(T)]      # [T x T]
    for row in range(T):  # Apply Softmax
        exps = [math.exp(v - max(scores[row])) for v in scores[row]]  # [1 x T] for each row
        attention[row] = [exps[col] / sum(exps) for col in range(T)]  # [1 x T] for each row

    return matmul(matmul(attention, V), Wo)   #[T x D]


def feed_forward(x, W1, b1, W2, b2):
    
    # x:  [T x D]
    # W1: [D x M]
    # b1: [1 x M]
    # W2: [M x D]
    # b2: [1 x D]
    
    T = len(x)              # scalar
    D = len(x[0])           # scalar
    M = len(b1)             # scalar

    h = matmul(x, W1)       # [T x M]
    for t in range(T): h[t] = [max(h[t][u] + b1[u], 0.0) for u in range(M)]
    h = matmul(h, W2)       # [T x D]
    for t in range(T): h[t] = [h[t][u] + b2[u] for u in range(D)]
    return h


def transformer_block(x, params):

    # x: [T x D]
    # params: parameter dictionary
    
    T = len(x)          # scalar
    D = len(x[0])       # scalarg

    gamma1 = params["ln1_gamma"]
    beta1  = params["ln1_beta"]
    Wq     = params["Wq"]
    Wk     = params["Wk"]
    Wv     = params["Wv"]
    Wout   = params["Wout"]
    gamma2 = params["ln2_gamma"]
    beta2  = params["ln2_beta"]
    W1     = params["W1"]
    b1     = params["b1"]
    W2     = params["W2"]
    b2     = params["b2"]

    ln1 = layernorm(x, gamma1 , beta1)                                  # [T x D]
    sa  = attention_head(ln1, Wq, Wk , Wv , Wout)                       # [T x D]
    res1 = [[x[t][d] + sa[t][d] for d in range(D)] for t in range(T)]   # [T x D]
    ln2 = layernorm(res1, gamma2, beta2)                                # [T x D]
    ff = feed_forward(ln2, W1, b1, W2, b2)                              # [T x D]
    out = [[res1[t][d] + ff[t][d] for d in range(D)] for t in range(T)] # [T x D]
    return out


def next_token_logits(tokens, model):
    
    # tokens: [T]
    # returns logits for last token: [V]
    
    T = len(tokens)        # scalar
    D = model["D"]         # scalar
    V = model["V"]         # scalar

    EE = model["token_emb"]
    PE = model["pos_emb"]
    block_params = model["blocks"]
    gamma_final = model["ln_f_gamma"]
    beta_final = model["ln_f_beta"]
    lm_head = model["lm_head"]

    x = [[0.0 for _ in range(D)] for _ in range(T)]             # [T x D]

    for t in range(T):
        tok_emb = EE[tokens[t]]                                 # [1 x D]
        pos_emb = PE[t]                                         # [1 x D]
        for d in range(D): x[t][d] = tok_emb[d] + pos_emb[d]    # scalar

    for blk in block_params: x = transformer_block(x, blk)      # [T x D]
    x = layernorm(x, gamma_final, beta_final)                   # [T x D]
    logits = matmul(x, lm_head)                                 # [T x V]
    return logits[-1]                                           # [V]


###############################################################
# Autoregressive generation
###############################################################

def generate(model, prompt_tokens, L):
    
    # prompt_tokens: [T_start]
    # Returns: [T_start + L]
    
    tokens = list(prompt_tokens)    # [T_current]
    V = model["V"]                  # scalar

    for _ in range(L):
        logits = next_token_logits(tokens, model)           # [1 x V]
        exps = [math.exp(v - max(logits)) for v in logits]  # [1 x V]
        probs = [e / sum(exps) for e in exps]               # [1 x V]
        next_tok = max(range(V), key=lambda i: probs[i])    # scalar
        tokens.append(next_tok)                             # [T_current+1]
    return tokens


###############################################################
# Random tiny model for testing
###############################################################

def random_vector(sz): return [random.uniform(-0.02, 0.02) for _ in range(sz)]                        # [1 x sz]
def random_matrix(m, n): return [[random.uniform(-0.02, 0.02) for _ in range(n)] for _ in range(m)]   # [m x n]

def build_toy_model(V=200, D=32, M=64, L=2, T_max=128):
    model = {
        "V": V,                             # vocab size
        "D": D,                             # embed dim
        "token_emb": random_matrix(V, D),   # [V, D]
        "pos_emb": random_matrix(T_max, D), # [T_max, D]
        "blocks": [],                       # list[L]
        "ln_f_gamma": random_vector(D),     # [D]
        "ln_f_beta": random_vector(D),      # [D]
        "lm_head": random_matrix(D, V)      # [D, V]
    }

    for _ in range(L):
        blk = {
            "Wq": random_matrix(D, D),          # [D x D]
            "Wk": random_matrix(D, D),          # [D x D]
            "Wv": random_matrix(D, D),          # [D x D]
            "Wout": random_matrix(D, D),        # [D x D]
            "ln1_gamma": random_vector(D),      # [1 x D]
            "ln1_beta": random_vector(D),       # [1 x D]
            "W1": random_matrix(D, M),          # [D x M]
            "b1": random_vector(M),             # [1 x M]
            "W2": random_matrix(M, D),          # [M x D]
            "b2": random_vector(D),             # [1 x D]
            "ln2_gamma": random_vector(D),      # [1 x D]
            "ln2_beta": random_vector(D)        # [1 x D]
        }
        model["blocks"].append(blk)

    return model


###############################################################
# Demo
###############################################################

if __name__ == "__main__":

    random.seed(0)
    model = build_toy_model()
    prompt = [10, 3, 17]
    result = generate(model, prompt, L=5)
    print("Generated:", result)
