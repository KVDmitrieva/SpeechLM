from abc import abstractmethod
from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F


class BaseFusion(nn.Module):
    """
    Base class for all fusion blocks
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x, y) -> float:
        raise NotImplementedError()
    
    @abstractmethod
    def predict(self, x) -> float:
        raise NotImplementedError()


class LinearFusion(BaseFusion):
    def __init__(self, input_dim=768, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def predict(self, x):
        x_proj = self.projection(x) # [batch, time, 1]
        return x_proj.mean(dim=[1, 2]) #[batch]

    def forward(self, x, y):
        x_pred = self.predict(x)
        y_pred = self.predict(y)

        out = 1. - F.sigmoid(y_pred - x_pred)
        return out


class CrossAttentionFusion(BaseFusion):
    def __init__(self, input_dim=768, hidden_dim=128, dropout=0.1):
        super().__init__()

        self.q_proj = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.k_proj = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.v_proj = nn.Linear(in_features=input_dim, out_features=hidden_dim)

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def predict(self, x, y=None, return_attention=False):
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x) if y is None else self.v_proj(y)

        res, attention = self._scaled_softmax_attention(query, key, value)
       
        proj = self.projection(res)
        out = 1. - F.sigmoid(proj)

        if return_attention:
            return out, attention
        
        return out

    def forward(self, x, y, return_attention=False):
        x, y = self._fix_shapes(x, y)
        return self.predict(x, y, return_attention)
    
    def _scaled_softmax_attention(self, query, key, value):
        scaled_dot = torch.matmul(query, key.transpose(-1, -2)) / sqrt(key.shape[-1])
        attention = nn.functional.softmax(scaled_dot, dim=-1)
        res = torch.matmul(attention, value)

        return res, attention
    
    def _fix_shapes(self, x, y):
        diff = y.shape[-1] - x.shape[-1]
        x = F.pad(x, (0, max(0, diff)))
        y = F.pad(y, (0, max(0, -diff)))
        return x, y