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
    def __init__(self, data_path: str, nb_subjects: int, cache_size: int = 1000):
        super().__init__()
        self.data_path = Path(data_path)
        self.cache_size = cache_size
        self.cache: OD[int, BaseConcatDataset] = OrderedDict()
        logging.info("Start initialization")
        self.subject_and_intervals(nb_subjects)
        assert hasattr(self, "_len")
        logging.info("Finish initialization")

    def subject_and_intervals(self, nb_subjects: int):
        subjects = []
        sizes = np.array([])
        bound = (nb_subjects // self.cache_size) + 1
        for start in tqdm(
            range(0, bound),
            "Prepare dataset",
        ):
            subject_ids = list(
                range(
                    start * self.cache_size,
                    min((start + 1) * self.cache_size, nb_subjects),
                )
            )
            subject_data = load_concat_dataset(
                self.data_path, preload=False, ids_to_load=subject_ids
            )
            subjects.extend(subject_ids)
            sub_sizes = np.array(subject_data.cumulative_sizes)
            sizes = np.concat([sizes, [sub_sizes[0]], sub_sizes[1:] - sub_sizes[:-1]])
        sizes = sizes.astype(np.int32)
        self._len = sizes.sum()
        self.subject = np.array(subjects)
        self.sizes = sizes

    def get_ids(self, index: int) -> Tuple[int, int]:
        upper = self.sizes.cumsum()
        lower = np.concat([np.zeros(1, dtype=np.int32), upper[:-1]])
        location = np.argwhere((lower <= index) & (index < upper))
        if location.size == 1:
            location = location.item()
        elif location.size == 0:
            raise StopIteration
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
            self.cache.popitem(last=False)

    def __len__(self):
        return self._len
