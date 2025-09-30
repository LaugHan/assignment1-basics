import torch
import torch.nn as nn
import math

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
