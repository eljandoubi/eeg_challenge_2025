import logging
import warnings
from collections.abc import Iterable
from pathlib import Path

from braindecode.datasets.base import BaseConcatDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    create_windows_from_events,
    preprocess,
)
from eegdash.dataset import EEGChallengeDataset
from eegdash.hbn.windows import (
    add_aux_anchors,
    add_extras_columns,
    annotate_trials_with_target,
    keep_only_recordings_with,
)
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore")

PASSIVE_TASKS = [
    "ThePresent",
    "FunwithFractals",
    "DespicableMe",
    "DiaryOfAWimpyKid",
    "surroundSupp",
    "RestingState",
]

ACTIVE_TASKS = [
    "symbolSearch",
    "seqLearning8target",
    "seqLearning6target",
    "contrastChangeDetection",
]

EPOCH_LEN_S = 2.0
SFREQ = 100  # by definition here

ANCHOR = "stimulus_anchor"

SHIFT_AFTER_STIM = 0.5
WINDOW_LEN = 2.0


def data_download(data_path: Path):
    for i in tqdm(range(1, 12), desc="R"):
        dataset = EEGChallengeDataset(
            task=PASSIVE_TASKS + ACTIVE_TASKS,
            release=f"R{i}",
            cache_dir=data_path,
            mini=False,
        )

        Parallel(n_jobs=-1)(
            delayed(lambda d: d.raw)(d) for d in tqdm(dataset.datasets, desc="ds")
        )


def process_ccd(inter: Iterable[int], cache_dir: str):
    dataset_ccd = BaseConcatDataset(
        [
            EEGChallengeDataset(
                task="contrastChangeDetection",
                release=f"R{i}",
                cache_dir=cache_dir,
                mini=False,
            )
            for i in tqdm(inter, desc="R")
        ]
    )

    transformation_offline = [
        Preprocessor(
            annotate_trials_with_target,
            target_field="rt_from_stimulus",
            epoch_length=EPOCH_LEN_S,
            require_stimulus=True,
            require_response=True,
            apply_on_array=False,
        ),
        Preprocessor(add_aux_anchors, apply_on_array=False),
    ]

    preprocess(dataset_ccd, transformation_offline, n_jobs=-1)

    # Keep only recordings that actually contain stimulus anchors
    dataset = keep_only_recordings_with(ANCHOR, dataset_ccd)

    # Create single-interval windows (stim-locked, long enough to include the response)
    single_windows = create_windows_from_events(
        dataset,
        mapping={ANCHOR: 0},
        trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),  # +0.5 s
        trial_stop_offset_samples=int(
            (SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ
        ),  # +2.5 s
        window_size_samples=int(EPOCH_LEN_S * SFREQ),
        window_stride_samples=SFREQ,
        preload=True,
        n_jobs=-1,
    )

    # Injecting metadata into the extra mne annotation.
    single_windows = add_extras_columns(
        single_windows,
        dataset,
        desc=ANCHOR,
        keys=(
            "target",
            "rt_from_stimulus",
            "rt_from_trialstart",
            "stimulus_onset",
            "response_onset",
            "correct",
            "response_type",
        ),
    )

    return single_windows


def parallel_filter(func, iterable, n_jobs=-1):
    # Run predicate in parallel
    mask = Parallel(n_jobs=n_jobs)(delayed(func)(item) for item in iterable)
    # Keep items where predicate returned True
    return [item for item, keep in zip(iterable, mask, strict=True) if keep]


def resolve_tasks(tasks: str | list[str]) -> str | list[str]:
    """Expand 'passive'/'active'/'all' shorthands and validate the task spec."""
    if isinstance(tasks, str):
        lowered = tasks.lower()
        if lowered == "passive":
            tasks = PASSIVE_TASKS
        elif lowered == "active":
            tasks = ACTIVE_TASKS
        elif lowered == "all":
            tasks = PASSIVE_TASKS + ACTIVE_TASKS

    if isinstance(tasks, str):
        assert tasks in PASSIVE_TASKS + ACTIVE_TASKS, (
            f"tasks must be valid task, got {tasks}"
        )
    elif isinstance(tasks, list):
        assert set(tasks).issubset(PASSIVE_TASKS + ACTIVE_TASKS), (
            f"tasks must be sub set of valid tasks, got {tasks}"
        )
    else:
        raise ValueError(f"tasks must be of type str or list, got {type(tasks)}")

    return tasks


def fixed_windows(
    tasks: str | list[str],
    inter: Iterable[int],
    cache_dir: str,
    wind: int = int(EPOCH_LEN_S * SFREQ),
):
    tasks = resolve_tasks(tasks)

    dss = []
    for i in tqdm(inter, desc="R"):
        try:
            ds = EEGChallengeDataset(
                task=tasks,
                release=f"R{i}",
                cache_dir=cache_dir,
                mini=False,
            )
            ds.datasets = parallel_filter(
                lambda d: d.raw.n_times >= wind, tqdm(ds.datasets, desc="Fitler wind")
            )
            dss.append(ds)
        except Exception as e:
            print(e)

    concat_ds = BaseConcatDataset(dss)
    logging.info("create fixed length windows")
    single_windows = create_fixed_length_windows(
        concat_ds,
        window_size_samples=wind,
        window_stride_samples=wind // 2,
        n_jobs=-1,
        drop_last_window=True,
    )

    return single_windows


if __name__ == "__main__":
    data_download("./data")
