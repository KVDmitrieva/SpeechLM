import torch

from src.metric.base_metric import BaseMetric


class MeanCleanScore(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def __call__(self, x_prediction, y_prediction, l_value, **batch):
        mask = l_value > 0.5
        mean_score = (x_prediction[mask].sum() + y_prediction[~mask].sum()) / l_value.shape[0]
        return mean_score
    

class MeanAugScore(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def __call__(self, x_prediction, y_prediction, l_value, **batch):
        mask = l_value < 0.5
        mean_score = (x_prediction[mask].sum() + y_prediction[~mask].sum()) / l_value.shape[0]
        return mean_score
    

class Accuracy(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def __call__(self, x_prediction, y_prediction, l_value, fusion_score, **batch):
        if x_prediction.dim() > 1:
            x_prediction = x_prediction.mean(dim=1).squeeze(-1)
            y_prediction = y_prediction.mean(dim=1).squeeze(-1)

        preds = torch.cat([x_prediction.unsqueeze(0), y_prediction.unsqueeze(0)], dim=0)
        labels = torch.min(preds, dim=0)[1]
        true_labels = l_value > 0.5

        mean_score = (true_labels == labels).float().mean()
        return mean_score
