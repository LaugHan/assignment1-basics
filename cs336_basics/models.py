import torch
import torch.nn as nn
import math
from einops import rearrange, einsum, reduce
import einx

class Embedding(nn.Module):
    def __init__(self, vocab_size:int, d_model:int, device:torch.device|None = None, dtype:torch.device|None = None):
        super().__init__()

        init_weights = torch.ones((vocab_size, d_model), device=device, dtype=dtype)
        weights = nn.init.trunc_normal_(init_weights, mean = 0, std = 1, a = 3, b = 3)

        self.weights = nn.Parameter(weights)

    def forward(self, token_ids:torch.Tensor) -> torch.Tensor:
        embedded_tokens = self.weights[token_ids]
        return embedded_tokens

class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device:torch.device|None = None, dtype:torch.dtype|None = None):
        super().__init__()

        init_weights = torch.rand((out_features, in_features), device=device, dtype=dtype)
        sigma_square = 2/(in_features + out_features)
        sigma = math.sqrt(sigma_square)
        weights = nn.init.trunc_normal_(init_weights, mean = 0, std = sigma, a = -3*sigma, b = 3*sigma)

        self.weights = nn.Parameter(weights)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... in, out  in -> ... out")
    

class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-5, device:torch.device|None = None, dtype:torch.dtype|None = None):
        super().__init__()
        
        self.eps = eps

        self.weights = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        RMS = torch.sqrt(reduce(torch.pow(x, 2), "b s d -> b s 1", "mean") + self.eps)
        norm = x / RMS * self.weights

        return norm.to(in_dtype)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff:int, device:torch.device|None = None, dtype:torch.dtype|None = None) -> None:
        super().__init__()

        W1 = torch.rand((d_ff, d_model), device=device, dtype=dtype)
        W2 = torch.rand((d_model, d_ff), device=device, dtype=dtype)
        W3 = torch.rand((d_ff, d_model), device=device, dtype=dtype)
        
        simga = math.sqrt(2/(d_ff + d_model))
        W1 = nn.init.trunc_normal_(W1, mean = 0, std = simga, a = -3*simga, b = 3*simga)
        W2 = nn.init.trunc_normal_(W2, mean = 0, std = simga, a = -3*simga, b = 3*simga)
        W3 = nn.init.trunc_normal_(W3, mean = 0, std = simga, a = -3*simga, b = 3*simga)

        self.W1 = nn.Parameter(W1)
        self.W2 = nn.Parameter(W2)
        self.W3 = nn.Parameter(W3)
    
    def SiLU(self, x:torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        gates = self.SiLU(x @ self.W1.T)
        linear = x @ self.W3.T
        return (gates * linear) @ self.W2.T
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len: int, device:torch.device|None = None):
        super().__init__()

        self.register_buffer(
            "_freq_cis_cache",
            RotaryPositionalEmbedding._init_cache(max_seq_len, d_k, theta), persistent=False
        )
    
    @staticmethod
    def _init_cache(max_seq_len:int, d_k:int, theta:float) -> torch.Tensor:
        assert d_k % 2 == 0

        d = torch.arange(0, d_k, 2) / d_k
        freq = theta ** -d
        t = torch.arange(max_seq_len)

        freqs = einsum(t, freq, "t, f -> t f") # [A]_{i, k}

        cos, sin = torch.cos(freqs), torch.sin(freqs) 
        return torch.stack((cos, sin))
    
    def forward(self, x:torch.Tensor, token_positions:torch.Tensor) -> torch.Tensor:
        x1, x2 = rearrange(x, "... (half_d xy) -> xy ... half_d", xy = 2) # 2 * [b s half_d]
        cos, sin = einx.get_at("cos_sin [pos] half_d, ... -> cos_sin ... half_d", self._freq_cis_cache, token_positions)  # 2 * [(b) s half_d]

        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        result = einx.rearrange( 
            "... half_d, ... half_d -> ... (half_d (1 + 1))",
            x1_rot, x2_rot
            ).contiguous()
        return result