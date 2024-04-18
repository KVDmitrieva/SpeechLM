import logging
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from src.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(self, index, config_parser: ConfigParser, limit=None, max_audio_length=None):
        self.config_parser = config_parser
        self.log_spec = config_parser["preprocessing"]["log_spec"]

        self.max_len = max_audio_length
        index = self._filter_records_from_dataset(index, limit)
        self._index: List[dict] = index

    def __getitem__(self, ind):
        data_dict = self._index[ind]

        clean_audio = self.load_audio(data_dict['clean_path'])
        aug_audio = self.load_audio(data_dict['aug_path'])

        return {
            "clean audio": clean_audio,
            "aug_audio": aug_audio
        }

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def process_wave(self, audio_tensor_wave: Tensor):
        with torch.no_grad():
            wave2spec = self.config_parser.init_obj(
                self.config_parser["preprocessing"]["spectrogram"],
                torchaudio.transforms,
            )
            audio_tensor_spec = wave2spec(audio_tensor_wave)
            if self.log_spec:
                audio_tensor_spec = torch.log(audio_tensor_spec + 1e-5)
            return audio_tensor_wave, audio_tensor_spec

    @staticmethod
    def _filter_records_from_dataset(index: list, limit) -> list:
        if limit is not None:
            random.seed(42)
            random.shuffle(index)
            index = index[:limit]
        return index

