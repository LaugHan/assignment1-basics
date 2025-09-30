import torch
import torch.nn as nn
import math
from einops import einsum

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