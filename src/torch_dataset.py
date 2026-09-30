from collections import OrderedDict
from collections import OrderedDict as OD
from gc import collect
from pathlib import Path

import numpy as np
from braindecode.datasets.base import BaseConcatDataset
from braindecode.datautil import load_concat_dataset
from torch.utils.data import Dataset
from tqdm import tqdm


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
        bound = (self.nb_subjects + self.cache_size - 1) // self.cache_size
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
            sizes = np.concatenate(
                [sizes, [sub_sizes[0]], sub_sizes[1:] - sub_sizes[:-1]]
            )

        return sizes.astype(np.int64), np.array(subjects, dtype=np.int64)

    def get_ids(self, index: int) -> tuple[int, int]:
        location = np.argwhere((self._lower <= index) & (index < self._upper))
        if location.size == 1:
            location = location.item()
        elif location.size == 0:
            raise IndexError(
                f"index {index} out of range for dataset of size {len(self)}"
            )
        else:
            raise ValueError("location must be unique")

        subject_id = self.subject[location].item()
        local_id = index - self._lower[location]
        return subject_id, local_id.item()

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        subject_id, local_id = self.get_ids(index)
        self.update_cache(subject_id)
        return self.cache[subject_id][local_id]

    def update_cache(self, subject_id: int):
        if subject_id in self.cache:
            self.cache.move_to_end(subject_id)
        else:
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
        # Precompute index bounds once instead of on every __getitem__ call.
        self._upper = sizes.cumsum()
        self._lower = np.concatenate([np.zeros(1, dtype=np.int64), self._upper[:-1]])
