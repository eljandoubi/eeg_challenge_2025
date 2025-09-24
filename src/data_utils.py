import warnings
from pathlib import Path
from typing import Iterable, Union, List
from tqdm import tqdm

from joblib import Parallel, delayed
from eegdash.dataset import EEGChallengeDataset
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
    create_windows_from_events,
    create_fixed_length_windows,
)
from braindecode.datasets.base import BaseConcatDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)

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


def fixed_windows(
    tasks: Union[str, List[str]],
    inter: Iterable[int],
    cache_dir: str,
    wind: int = int(EPOCH_LEN_S * SFREQ),
):
    if tasks.lower() == "passive":
        tasks = PASSIVE_TASKS
    elif tasks.lower() == "active":
        tasks = ACTIVE_TASKS
    elif tasks.lower() == "all":
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
        ValueError(f"tasks must be of type str or list, got {type(tasks)}")

    dss = []
    for i in tqdm(inter, desc="R"):
        try:
            ds = EEGChallengeDataset(
                task=tasks,
                release=f"R{i}",
                cache_dir=cache_dir,
                mini=False,
            )
            ds.datasets = [
                d
                for d in tqdm(ds.datasets, desc="Fitler wind")
                if d.raw.n_times >= wind
            ]
            dss.append(ds)
        except Exception as e:
            print(e)

    concat_ds = BaseConcatDataset(dss)

    single_windows = create_fixed_length_windows(
        concat_ds,
        window_size_samples=wind,
        window_stride_samples=wind // 2,
        preload=True,
        n_jobs=-1,
        drop_last_window=True,
        on_missing="warning",
    )

    return single_windows


if __name__ == "__main__":
    data_download()
    # train_inter = set(range(1, 11))
    # train_inter.remove(5)
    # train_inter = [5]
    # print(train_inter)
    # sw = process_ccd(train_inter)
    # sw = fixed_windows(tasks="passive",inter=train_inter)

    # meta_information = sw.get_metadata()
    # print(meta_information.columns)
    # subjects = meta_information["subject"].unique()
    # print(subjects)
    # for i, x in enumerate(sw):
    #     print(x)
    #     print(len(x))
    #     if i == 5:
    #         break
