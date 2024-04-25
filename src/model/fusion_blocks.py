from abc import abstractmethod

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

