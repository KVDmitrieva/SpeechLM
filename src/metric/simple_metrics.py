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