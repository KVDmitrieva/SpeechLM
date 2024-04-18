import torch
import torch.nn as nn


class RankLoss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, clean_score, aug_score, **batch):
        p = torch.exp(clean_score - aug_score) / (1 + torch.exp(clean_score - aug_score))
        L_rank = - torch.log(p)
        return {
            "loss": L_rank
        }
