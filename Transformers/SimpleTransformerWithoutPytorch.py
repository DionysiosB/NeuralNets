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
        stdev = math.sqrt(sum([u ** 2 for u in x[t]]) / D - mean ** 2) + eps   # Add epsilon to avoid potential zero division
        for d in range(D): y[t][d] = gamma[d] * (x[t][d] - mean) / stdev + beta[d]

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

    scores = [[-1e9 for _ in range(T)] for _ in range(T)]    # [T x T]

    for row in range(T):
        for col in range(T):
            if row < col: continue    # masked - Rows are the queries, columns the keys
            s = 0.0
            for d in range(D): s += Q[row][d] * K[col][d]
            scores[row][col] = s / math.sqrt(D)

    attention = [[0.0 for _ in range(T)] for _ in range(T)]      # [T x T]
    for row in range(T):  # Apply Softmax
        exps = [math.exp(v - max(scores[row])) for v in scores[row]]  # [1 x T] for each row
        attention[row] = [exps[col] / sum(exps) for col in range(T)]  # [1 x T] for each row

    return matmul(matmul(attention, V), Wo)   #[T x D]


def feed_forward(x, Wup, bup, Wdown, bdown):
    
    # x:     [T x D]
    # Wup:   [D x M]
    # bup:   [1 x M]
    # Wdown: [M x D]
    # bdown: [1 x D]
    
    T = len(x)              # scalar
    D = len(x[0])           # scalar
    M = len(bup)            # scalar

    h = matmul(x, Wup)      # [T x M]
    for t in range(T): h[t] = [max(h[t][u] + bup[u], 0.0) for u in range(M)]  #ReLU
    h = matmul(h, Wdown)    # [T x D]
    for t in range(T): h[t] = [h[t][u] + bdown[u] for u in range(D)]
    return h


def transformer_block(x, params):

    # x: [T x D]
    # params: parameter dictionary
    
    T = len(x)          # scalar
    D = len(x[0])       # scalar

    gamma1 = params["gamma1"]
    beta1  = params["beta1"]
    Wq     = params["Wq"]
    Wk     = params["Wk"]
    Wv     = params["Wv"]
    Wout   = params["Wout"]
    gamma2 = params["gamma2"]
    beta2  = params["beta2"]
    Wup    = params["Wup"]
    bup    = params["bup"]
    Wdown  = params["Wdown"]
    bdown  = params["bdown"]

    ln1  = layernorm(x, gamma1 , beta1)                                    # [T x D]
    sab  = attention_head(ln1, Wq, Wk , Wv , Wout)                         # [T x D]
    res1 = [[x[t][d] + sab[t][d] for d in range(D)] for t in range(T)]     # [T x D]
    ln2  = layernorm(res1, gamma2, beta2)                                  # [T x D]
    ffn  = feed_forward(ln2, Wup, bup, Wdown, bdown)                       # [T x D]
    out  = [[res1[t][d] + ffn[t][d] for d in range(D)] for t in range(T)]  # [T x D]
    return out


def next_token_logits(tokens, model):
    
    # tokens: [1 x T]
    # returns logits for last token: [1 x V]
    
    T = len(tokens)        # scalar
    D = model["D"]         # scalar
    V = model["V"]         # scalar

    EE = model["token_emb"]
    PE = model["pos_emb"]
    block_params = model["blocks"]
    gamma_final = model["gamma"]
    beta_final = model["beta"]
    lm_head = model["lm_head"]

    x = [[0.0 for _ in range(D)] for _ in range(T)]                    # [T x D]

    for t in range(T):
        tok_emb = EE[tokens[t]]                                        # [1 x D]
        pos_emb = PE[t]                                                # [1 x D]
        for d in range(D): x[t][d] = tok_emb[d] + pos_emb[d]           # scalar

    for blk in block_params: x = transformer_block(x, blk)             # [T x D]
    logits = matmul(layernorm(x, gamma_final, beta_final), lm_head)    # [T x V]
    return logits[-1]                                                  # [1 x V]


def generate(model, prompt_tokens, L):
    
    # prompt_tokens: [1 x T_start]
    # Returns: [1 x (T_start + L)]
    
    tokens = list(prompt_tokens)    # [1 x T_current]
    V = model["V"]                  # scalar

    for _ in range(L):
        logits = next_token_logits(tokens, model)           # [1 x V]
        exps = [math.exp(v - max(logits)) for v in logits]  # [1 x V]
        probs = [e / sum(exps) for e in exps]               # [1 x V]
        next_tok = max(range(V), key=lambda i: probs[i])    # scalar
        tokens.append(next_tok)                             # [1 x (T_current+1)]
    return tokens


###############################################################
##################### Demo ####################################
###############################################################

def random_vector(sz): return [random.uniform(-0.02, 0.02) for _ in range(sz)]                        # [1 x sz]
def random_matrix(m, n): return [[random.uniform(-0.02, 0.02) for _ in range(n)] for _ in range(m)]   # [m x n]

def build_toy_model(V=200, D=32, M=64, L=2, T_max=128):
    model = {
        "V": V,                             # vocab size
        "D": D,                             # embed dim
        "token_emb": random_matrix(V, D),   # [V x D]
        "pos_emb": random_matrix(T_max, D), # [T_max x D]
        "blocks": [],                       # [1 x L]
        "gamma": random_vector(D),          # [1 x D]
        "beta": random_vector(D),           # [1 x D]
        "lm_head": random_matrix(D, V)      # [D x V]
    }

    for _ in range(L):
        blk = {
            "Wq": random_matrix(D, D),          # [D x D]
            "Wk": random_matrix(D, D),          # [D x D]
            "Wv": random_matrix(D, D),          # [D x D]
            "Wout": random_matrix(D, D),        # [D x D]
            "gamma1": random_vector(D),         # [1 x D]
            "beta1": random_vector(D),          # [1 x D]
            "Wup": random_matrix(D, M),         # [D x M]
            "bup": random_vector(M),            # [1 x M]
            "Wdown": random_matrix(M, D),       # [M x D]
            "bdown": random_vector(D),          # [1 x D]
            "gamma2": random_vector(D),         # [1 x D]
            "beta2": random_vector(D)           # [1 x D]
        }
        model["blocks"].append(blk)

    return model


if __name__ == "__main__":

    random.seed(0)
    model  = build_toy_model()
    prompt = [11, 19, 3]
    result = generate(model, prompt, L=5)
    print("Generated:", result)
