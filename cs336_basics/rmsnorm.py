import torch
import torch.nn as nn
from einops import reduce
import math

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