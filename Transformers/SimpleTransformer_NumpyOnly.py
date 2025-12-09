import numpy as np

def layernorm(x, gamma, beta, eps=1e-5):
    # x:     [T x D]
    # gamma: [1 x D]
    # beta:  [1 x D]
    # returns: [T x D]
    mean  = np.mean(x, axis=1, keepdims=True)           #[T x 1]
    stdev = np.std(x,  axis=1, keepdims=True) + eps     #[T x 1]
    return beta + gamma * (x - mean) / stdev            #[T x D]


def attention_head(x, Wq, Wk, Wv, Wo):
    # x:  [T x D]
    # Wq: [D x D]
    # Wk: [D x D]
    # Wv: [D x D]
    # Wo: [D x D]
    # Returns: [T x D]
    T, D = x.shape
    Q = np.matmul(x, Wq)   # [T x D]
    K = np.matmul(x, Wk)   # [T x D]
    V = np.matmul(x, Wv)   # [T x D]
    
    scores = np.matmul(Q, np.transpose(K)) / np.sqrt(D)                 # [T x T]
    scores[np.triu_indices(T, k=1)] = -np.inf             # Set upper triangle (excluding diagonal) to -inf   
    attention = np.exp(scores - np.max(scores, axis=1, keepdims=True))  # [T x T]
    attention /= np.sum(attention, axis=1, keepdims=True)
    
    return np.matmul(attention, V)                                      # [T x D]


def feed_forward(x, Wup, bup, Wdown, bdown):
    # x:     [T x D]
    # Wup:   [D x M]
    # bup:   [1 x M]
    # Wdown: [M x D]
    # bdown: [1 x D]
    # returns: [T x D]
    h = np.maximum(np.matmul(x, Wup) + bup, 0)  # ReLU activation
    h = np.matmul(h, Wdown) + bdown
    return h

def transformer_block(x, params):
    # x: [T x D]
    # params: parameter dictionary
    # Returns: [T x D]

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


    ln1  = layernorm(x, gamma1, beta1)                   # [T x D]
    sab  = attention_head(ln1, Wq, Wk, Wv, Wout)         # [T x D]
    res1 = x + sab                                       # [T x D]
    ln2  = layernorm(res1, gamma2, beta2)                # [T x D]
    ffn  = feed_forward(ln2, Wup, bup, Wdown, bdown)     # [T x D]
    return res1 + ffn                                    # [T x D]

def next_token_logits(tokens, model):
    # tokens: [1 x T]
    # returns logits for last token: [1 x V]
    T = len(tokens)
    D = model["D"]
    V = model["V"]
    EE = model["token_emb"]
    PE = model["pos_emb"]
    blocks = model["blocks"]
    gamma_final = model["gamma_final"]
    beta_final = model["beta_final"]
    lm_head = model["lm_head"]

    x = np.zeros((T, D))                                                # [T x D]
    for t in range(T): x[t] = EE[tokens[t]] + PE[t]                     # [1 x D]
    for blk in blocks: x = transformer_block(x, blk)                    # [T x D]
    logits = np.matmul(layernorm(x, gamma_final, beta_final), lm_head)  # [T x V]
    return logits[-1]                                                   # [1 x V]

def generate(model, prompt_tokens, L):
    # prompt_tokens: [1 x T_start]
    # Returns: [1 x (T_start + L)]
    tokens = list(prompt_tokens)  # [1 x T_current]
    V = model["V"]  # vocabulary size - scalar

    for _ in range(L):
        logits = next_token_logits(tokens, model)       # [1 x V]
        probs = np.exp(logits - np.max(logits))         # [1 x V]
        probs /= np.sum(probs)                          # Normalize to get probabilities
        next_tok = np.argmax(probs)                     # scalar
        tokens.append(next_tok)                         # [1 x (T_current + 1)]
    
    return tokens

###############################################################
##################### Demo ####################################
###############################################################

def random_vector(sz): return np.random.uniform(-0.02, 0.02, sz)  # [1 x sz]
def random_matrix(m, n): return np.random.uniform(-0.02, 0.02, (m, n))  # [m x n]

def build_toy_model(V=200, D=32, M=64, L=2, T_max=128):
    model = {
        "V": V,                             # vocab size
        "D": D,                             # embed dim
        "token_emb": random_matrix(V, D),   # [V x D]
        "pos_emb": random_matrix(T_max, D), # [T_max x D]
        "blocks": [],                       # [1 x L]
        "gamma_final": random_vector(D),    # [1 x D]
        "beta_final": random_vector(D),     # [1 x D]
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

    np.random.seed(0)
    model  = build_toy_model()
    prompt = [17, 11, 74]
    result = generate(model, prompt, L=5)
    print("Generated:", result)
