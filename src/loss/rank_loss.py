import torch
import torch.nn as nn


class RankLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fusion_score: torch.Tensor, l_value: torch.Tensor ,**batch):
        L_rank = - l_value * torch.log(fusion_score) - (1. - l_value) * torch.log(1. - fusion_score)
        return {
            "loss": L_rank
        }
