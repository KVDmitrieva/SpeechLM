import logging
import torch
from typing import List
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    clean_audio = []
    aug_audio = []


    for item in dataset_items:
        clean_audio.append(item["clean_audio"].T)
        aug_audio.append(item["aug_audio"].T)


    return {
        "aug_audio": pad_sequence(aug_audio, batch_first=True).transpose(1, 2),
        "clean_audio": pad_sequence(clean_audio, batch_first=True).transpose(1, 2)
    }