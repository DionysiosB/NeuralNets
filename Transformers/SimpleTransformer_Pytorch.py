import torch
import torch.nn.functional as F
from typing import List, Dict, Any


def layernorm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # x:     [T x D]
    # gamma: [1 x D] or [D]
    # beta:  [1 x D] or [D]
    mean = x.mean(dim=1, keepdim=True)                          # [T x 1]
    var  = x.var(dim=1, unbiased=False, keepdim=True)           # [T x 1]
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta    # [T x D]


def attention_head(x: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor, Wv: torch.Tensor, Wo: torch.Tensor) -> torch.Tensor:
    # x:    [T x D]
    # Wq:   [D x D]
    # Wk:   [D x D]
    # Wv:   [D x D]
    # Wout: [D x D]
    T, D = x.shape
    Q = torch.matmul(x, Wq)          # [T x D]
    K = torch.matmul(x, Wk)          # [T x D]
    V = torch.matmul(x, Wv)          # [T x D]

    scores = torch.matmul(Q, torch.transpose(K, 1, 0)) / (D ** 0.5)     # [T x T]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)   # [T x T]
    scores = scores.masked_fill(mask, float("-inf"))                    # [T x T] , causal masking
    attention = F.softmax(scores, dim=1)                                # [T x T]
    out = torch.matmul(attention, V)                                    # [T x D]
    return torch.matmul(out, Wo)                                        # [T x D]


def feed_forward(x: torch.Tensor, Wup: torch.Tensor, bup: torch.Tensor, Wdown: torch.Tensor, bdown: torch.Tensor) -> torch.Tensor:
    #x:     [T x D]
    #Wup:   [D x M]
    #bup:   [1 x M] or [M]
    #Wdown: [M x D]
    #bdown: [1 x D] or [D]
    h = F.relu(torch.add(torch.matmul(x, Wup) , bup))       # [T x M]
    return torch.add(torch.matmul(h, Wdown), bdown)         # [T x D]


def transformer_block(x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    # x: [T x D]
    # params: Parameter dictionary
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

    ln1  = layernorm(x, gamma1, beta1)              # [T x D]
    sab  = attention_head(ln1, Wq, Wk, Wv, Wout)    # [T x D]
    mid = x + sab                                   # [T x D]
    ln2 = layernorm(mid, gamma2, beta2)             # [T x D]
    ffn = feed_forward(ln2, Wup, bup, Wdown, bdown) # [T x D]
    return mid + ffn                                # [T x D]


def next_token_logits(tokens: List[int], model: Dict[str, Any]) -> torch.Tensor:    
    # tokens: [1 x T] as a python list of ints
    # returns logits for last token: [1 x V]
    T = len(tokens)
    D = model["D"]   # scalar
    V = model["V"]   # scalar
    EE = model["token_emb"]
    PE = model["pos_emb"]
    blocks = model["blocks"]
    gamma_final = model["gamma_final"]
    beta_final = model["beta_final"]
    lm_head = model["lm_head"] 

    x = EE[tokens] + PE[torch.arange(T)]               # [T x D]
    for blk in blocks: x = transformer_block(x, blk)   # [T x D]
    x = layernorm(x, gamma_final, beta_final)          # [T x D]
    logits = torch.matmul(x, lm_head)                  # [T x V]
    return logits[-1]                                  # [1 x V]


def generate(model: Dict[str, Any], prompt_tokens: List[int], L: int) -> List[int]:
    # prompt_tokens: [1 x T_start]
    # Returns: [1 x (T_start + L)]
    
    tokens = list(prompt_tokens)   # python list of ints
    V = model["V"]

    for _ in range(L):
        logits = next_token_logits(tokens, model)   # [1 x V]
        next_tok = torch.argmax(logits).item()      # scalar
        tokens.append(next_tok)                     # extend sequence
    return tokens


###############################################################
##################### Demo ####################################
###############################################################

def build_toy_model(V: int = 200, D: int = 32, M: int = 64, L: int = 2, T_max: int = 128) -> Dict[str, Any]:
    rnd = lambda *s: (torch.rand(*s) * 0.04 - 0.02)

    model: Dict[str, Any] = {
        "V": V,
        "D": D,
        "token_emb": rnd(V, D),     # [V x D]
        "pos_emb": rnd(T_max, D),   # [T_max x D], should ideally be sine/cosine functions 
        "gamma_final": rnd(D),      # [1 x D]
        "beta_final": rnd(D),       # [1 x D]
        "lm_head": rnd(D, V),       # [D x V]
        "blocks": []
    }

    for _ in range(L):
        blk = {
            "Wq": rnd(D, D),        # [D x D]
            "Wk": rnd(D, D),        # [D x D]
            "Wv": rnd(D, D),        # [D x D]
            "Wout": rnd(D, D),      # [D x D]
            "gamma1": rnd(D),       # [1 x D]
            "beta1": rnd(D),        # [1 x D]
            "Wup": rnd(D, M),       # [D x M]
            "bup": rnd(M),          # [1 x M]
            "Wdown": rnd(M, D),     # [M x D]
            "bdown": rnd(D),        # [1 x D]
            "gamma2": rnd(D),       # [1 x D]
            "beta2": rnd(D)         # [1 x D]
        }
        model["blocks"].append(blk)

    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    model  = build_toy_model()
    prompt = [11, 19, 3]
    result = generate(model, prompt, L=5)
    print("Generated:", result)
