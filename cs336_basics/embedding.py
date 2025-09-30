import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size:int, d_model:int, device:torch.device|None = None, dtype:torch.device|None = None):
        super().__init__()

        init_weights = torch.ones((vocab_size, d_model), device=device, dtype=dtype)
        weights = nn.init.trunc_normal_(init_weights, mean = 0, std = 1, a = 3, b = 3)

        self.weights = nn.Parameter(weights)

    def forward(self, token_ids:torch.Tensor) -> torch.Tensor:
        embedded_tokens = self.weights[token_ids]
        return embedded_tokens
