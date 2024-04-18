import logging
import json
from pathlib import Path

import numpy as np

from src.datasets.base_dataset import BaseDataset
from src.utils import ROOT_PATH
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FastspeechDataset(BaseDataset):
    """"
    train.json / test.json:

        text: str --- original text
        modified_text: str --- '.' -> ',' and random additional ','
        clean path: str --- path to clean audio
        aug path: str --- path to augmented audio
        clean durations: list[int]
        aug durations: list[int]
        phoneme changes: dict --- multipliers used for specific phonemes


    """
    def __init__(self, part, data_dir, *args, **kwargs):
        assert part in ["train", "test"]
        self._data_dir = Path(data_dir)
        self._index_dir = ROOT_PATH / "data" / "datasets" / "fastspeech"
        self._index_dir.mkdir(exist_ok=True, parents=True)
        index = self._get_or_load_index(part)
        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, part):
        index_path = self._index_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        data_json = self._data_dir / f"{part}.json"
        with open(data_json, "r") as f:
            files = json.load(f)

        for file in tqdm(files, desc=f"Prepeare {part} files"):
            clean_path = self._data_dir / file["clean path"]
            aug_path = self._data_dir / file["aug path"]

            index.append(
                {
                    "clean_path": str(clean_path.absolute().resolve()),
                    "aug_path": str(aug_path.absolute().resolve())
                }
            )
        return index

