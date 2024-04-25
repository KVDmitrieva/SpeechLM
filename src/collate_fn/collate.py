import logging
import torch
from typing import List
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    x, y, l_value = [], [], []


    for item in dataset_items:
        x.append(item["x"].T)
        y.append(item["y"].T)
        l_value.append(item["l_value"])


    return {
        "x": pad_sequence(x, batch_first=True).transpose(1, 2),
        "y": pad_sequence(y, batch_first=True).transpose(1, 2),
        "l_value": torch.tensor(l_value)
    }