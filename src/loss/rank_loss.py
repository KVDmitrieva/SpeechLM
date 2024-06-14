import torch
import torch.nn as nn


class RankLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-9

    def forward(self, fusion_score: torch.Tensor, l_value: torch.Tensor ,**batch):
        L_rank = - l_value * torch.log(fusion_score + self.eps) - (1. - l_value) * torch.log(1. - fusion_score + self.eps)
        return {
            "loss": L_rank.mean()
        }
