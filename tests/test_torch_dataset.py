import numpy as np
import pytest

import torch_dataset as torch_dataset_module
from torch_dataset import SubjectsDataset


def make_dataset(sizes, cache_size: int = 1000) -> SubjectsDataset:
    ds = SubjectsDataset(
        data_path="unused", nb_subjects=len(sizes), cache_size=cache_size
    )
    ds.syn_data(
        sizes=np.array(sizes, dtype=np.int64),
        subjects=np.arange(len(sizes), dtype=np.int64),
    )
    return ds


def test_len_matches_total_number_of_samples():
    ds = make_dataset([3, 5, 2])
    assert len(ds) == 10


def test_get_ids_maps_index_to_subject_and_local_id():
    ds = make_dataset([3, 5, 2])  # subject 0: 0-2, subject 1: 3-7, subject 2: 8-9

    assert ds.get_ids(0) == (0, 0)
    assert ds.get_ids(2) == (0, 2)
    assert ds.get_ids(3) == (1, 0)
    assert ds.get_ids(7) == (1, 4)
    assert ds.get_ids(8) == (2, 0)
    assert ds.get_ids(9) == (2, 1)


def test_get_ids_out_of_range_raises_index_error():
    """Regression test: this used to raise StopIteration, which is wrong here."""
    ds = make_dataset([3, 5, 2])

    with pytest.raises(IndexError):
        ds.get_ids(10)


def test_subject_and_intervals_has_no_trailing_empty_chunk(monkeypatch):
    """Regression test: an exact multiple of cache_size used to request one
    extra, empty chunk from load_concat_dataset."""
    calls = []

    class FakeConcatDataset:
        def __init__(self, ids_to_load):
            n = len(ids_to_load)
            self.cumulative_sizes = list(range(1, n + 1))

    def fake_load_concat_dataset(data_path, preload, ids_to_load):
        calls.append(list(ids_to_load))
        return FakeConcatDataset(ids_to_load)

    monkeypatch.setattr(
        torch_dataset_module, "load_concat_dataset", fake_load_concat_dataset
    )

    ds = SubjectsDataset(data_path="unused", nb_subjects=1000, cache_size=1000)
    _, subjects = ds.subject_and_intervals()

    assert len(calls) == 1
    assert all(len(chunk) > 0 for chunk in calls)
    assert subjects.tolist() == list(range(1000))


def test_subject_and_intervals_chunks_a_non_exact_multiple(monkeypatch):
    calls = []

    class FakeConcatDataset:
        def __init__(self, ids_to_load):
            n = len(ids_to_load)
            self.cumulative_sizes = list(range(1, n + 1))

    def fake_load_concat_dataset(data_path, preload, ids_to_load):
        calls.append(list(ids_to_load))
        return FakeConcatDataset(ids_to_load)

    monkeypatch.setattr(
        torch_dataset_module, "load_concat_dataset", fake_load_concat_dataset
    )

    ds = SubjectsDataset(data_path="unused", nb_subjects=2500, cache_size=1000)
    ds.subject_and_intervals()

    assert [len(chunk) for chunk in calls] == [1000, 1000, 500]


def test_update_cache_is_lru(monkeypatch):
    loaded = []

    def fake_load_concat_dataset(data_path, preload, ids_to_load):
        loaded.append(ids_to_load[0])
        return object()

    monkeypatch.setattr(
        torch_dataset_module, "load_concat_dataset", fake_load_concat_dataset
    )

    ds = SubjectsDataset(data_path="unused", nb_subjects=3, cache_size=2)
    ds.update_cache(0)
    ds.update_cache(1)
    ds.update_cache(0)  # touch subject 0 again -> becomes most recently used
    ds.update_cache(2)  # eviction: subject 1 (least recently used) should go

    assert list(ds.cache.keys()) == [0, 2]
    assert loaded == [0, 1, 2]  # subject 0 was not reloaded on the second touch


def test_getitem_supports_negative_indexing(monkeypatch):
    class FakeConcatDataset:
        def __init__(self, subject_id):
            self.subject_id = subject_id

        def __getitem__(self, local_id):
            return (self.subject_id, local_id)

    load_calls = []

    def fake_load_concat_dataset(data_path, preload, ids_to_load):
        load_calls.append(ids_to_load[0])
        return FakeConcatDataset(ids_to_load[0])

    monkeypatch.setattr(
        torch_dataset_module, "load_concat_dataset", fake_load_concat_dataset
    )

    ds = make_dataset([3, 5, 2], cache_size=10)

    assert ds[0] == (0, 0)
    assert ds[-1] == (2, 1)
    assert load_calls == [0, 2]
