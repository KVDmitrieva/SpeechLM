import torch

from src.metric.base_metric import BaseMetric


class MeanCleanScore(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def __call__(self, x_prediction, y_prediction, l_value, fusion_score, **batch):
        mask = l_value > 0.5
        mean_score = fusion_score[mask].mean()
        return mean_score
    

class MeanAugScore(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def __call__(self, x_prediction, y_prediction, l_value, fusion_score, **batch):
        mask = l_value < 0.5
        mean_score = fusion_score[mask].mean()
        return mean_score
    

class Accuracy(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def __call__(self, x_prediction, y_prediction, l_value, fusion_score, **batch):
        # preds = torch.cat([x_prediction.unsqueeze(0), y_prediction.unsqueeze(0)], dim=0)
        labels = fusion_score >= 0.5
        true_labels = l_value >= 0.5

        mean_score = (true_labels == labels).float().mean()
        return mean_score
