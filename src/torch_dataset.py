from gc import collect
from pathlib import Path
from collections import OrderedDict
from typing import Tuple, OrderedDict as OD

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from braindecode.datautil import load_concat_dataset
from braindecode.datasets.base import BaseConcatDataset


class SubjectsDataset(Dataset):
    def __init__(self, data_path: str, nb_subjects: int, cache_size: int = 1000):
        super().__init__()
        self.data_path = Path(data_path)
        self.cache_size = cache_size
        self.nb_subjects = nb_subjects
        self.cache: OD[int, BaseConcatDataset] = OrderedDict()

    def subject_and_intervals(self):
        subjects = []
        sizes = np.array([])
        bound = (self.nb_subjects // self.cache_size) + 1
        for start in tqdm(
            range(0, bound),
            "Prepare dataset",
        ):
            subject_ids = list(
                range(
                    start * self.cache_size,
                    min((start + 1) * self.cache_size, self.nb_subjects),
                )
            )
            subject_data = load_concat_dataset(
                self.data_path, preload=False, ids_to_load=subject_ids
            )
            subjects.extend(subject_ids)
            sub_sizes = np.array(subject_data.cumulative_sizes)
            sizes = np.concat([sizes, [sub_sizes[0]], sub_sizes[1:] - sub_sizes[:-1]])

        return sizes.astype(np.long), np.array(subjects, dtype=np.long)

    def get_ids(self, index: int) -> Tuple[int, int]:
        upper = self.sizes.cumsum()
        lower = np.concat([np.zeros(1, dtype=np.long), upper[:-1]])
        location = np.argwhere((lower <= index) & (index < upper))
        if location.size == 1:
            location = location.item()
        elif location.size == 0:
            raise StopIteration("index out of range")
        else:
            raise ValueError("location must be unique")

        subject_id = self.subject[location].item()
        local_id = index - lower[location]
        return subject_id, local_id.item()

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        subject_id, local_id = self.get_ids(index)
        self.update_cache(subject_id)
        return self.cache[subject_id][local_id]

    def update_cache(self, subject_id: int):
        if subject_id not in self.cache:
            self.cache[subject_id] = load_concat_dataset(
                self.data_path, preload=False, ids_to_load=[subject_id]
            )
        if len(self.cache) > self.cache_size:
            _, ds = self.cache.popitem(last=False)
            del ds
            collect()

    def __len__(self):
        return self._len

    def syn_data(self, sizes: np.ndarray, subjects: np.ndarray):
        self.subject = subjects
        self.sizes = sizes
        self._len = sizes.sum()
