import logging
from pathlib import Path
from typing import Tuple, OrderedDict as OD
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from braindecode.datautil import load_concat_dataset
from braindecode.datasets.base import BaseConcatDataset


class SubjectsDataset(Dataset):
    def __init__(self, data_path: str, cache_size: int = 1000):
        super().__init__()
        self.data_path = Path(data_path)
        self.cache_size = cache_size
        self.cache: OD[int, BaseConcatDataset] = OrderedDict()
        logging.info("Start initialization")
        self.subject_and_intervals()
        assert hasattr(self, "_len")
        logging.info("Finish initialization")

    def subject_and_intervals(self):
        S = 0
        subjects = []
        lower = []
        upper = []
        for p in tqdm(
            self.data_path.iterdir(),
            "Prepare dataset",
            sum(1 for _ in self.data_path.iterdir()),
        ):
            folder = p.name
            if p.is_dir() and folder.isdigit():
                subject_id = int(str(folder))
                subject_data = load_concat_dataset(
                    self.data_path, preload=True, ids_to_load=[subject_id]
                )
                ln = len(subject_data)
                subjects.append(subject_id)
                lower.append(S)
                upper.append(S + ln)
                S += ln
        self._len = S
        self.subject = np.array(subjects)
        self.lower = np.array(lower)
        self.upper = np.array(upper)

    def get_ids(self, index: int) -> Tuple[int, int]:
        location = np.argwhere((self.lower <= index) & (index < self.upper))
        try:
            location = location.item()
        except Exception:
            location = -1

        subject_id = self.subject[location].item()
        local_id = index - self.lower[location]
        return subject_id, local_id.item()

    def __getitem__(self, index: int):
        subject_id, local_id = self.get_ids(index)
        self.update_cache(subject_id)
        return self.cache[subject_id][local_id]

    def update_cache(self, subject_id: int):
        if subject_id not in self.cache:
            self.cache[subject_id] = load_concat_dataset(
                self.data_path, preload=True, ids_to_load=[subject_id]
            )
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def __len__(self):
        return self._len
