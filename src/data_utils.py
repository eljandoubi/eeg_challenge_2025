import warnings
from pathlib import Path
from typing import Iterable
from tqdm import tqdm  # noqa: F401

from joblib import Parallel, delayed
from eegdash.dataset import EEGChallengeDataset
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
    create_windows_from_events,
)
from braindecode.datasets.base import BaseConcatDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)

warnings.filterwarnings("ignore")

DATA_DIR = Path("/home/a/startkit/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

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
    "contrastChangeDetection",
]

EPOCH_LEN_S = 2.0
SFREQ = 100  # by definition here

ANCHOR = "stimulus_anchor"

SHIFT_AFTER_STIM = 0.5
WINDOW_LEN = 2.0


def data_download(data_path: Path = DATA_DIR):
    for i in tqdm(range(1, 12), desc="R"):
        dataset = EEGChallengeDataset(
            task=[
                "ThePresent",
                "FunwithFractals",
                "DespicableMe",
                "DiaryOfAWimpyKid",
                "symbolSearch",
                "contrastChangeDetection",
                "surroundSupp",
                "RestingState",
                "seqLearning8target",
            ],
            release=f"R{i}",
            cache_dir=data_path,
            mini=False,
        )

        Parallel(n_jobs=-1)(
            delayed(lambda d: d.raw)(d) for d in tqdm(dataset.datasets, desc="ds")
        )


def process_ccd(inter: Iterable[int] = range(1, 12), cache_dir: str = DATA_DIR):
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


if __name__ == "__main__":
    # data_download()
    # train_inter = set(range(1, 11))
    # train_inter.remove(5)
    train_inter = [5]
    print(train_inter)
    # dataset = BaseConcatDataset(
    #     [
    #         EEGChallengeDataset(
    #             task="DespicableMe",
    #             release=f"R{i}",
    #             cache_dir=DATA_DIR,
    #             mini=False,
    #         )
    #         for i in tqdm(train_inter, desc="R")
    #     ]
    # )
    # print(dataset.datasets[0].raw.to_data_frame().head())

    # for t in PASSIVE_TASKS+ACTIVE_TASKS:
    #     print("task",t)
    #     eeg = EEGChallengeDataset(
    #             task=t,
    #             release="R5",
    #             cache_dir=DATA_DIR,
    #             mini=True,
    #         )
    # print("target", len(eeg.description.columns))

    # for d in dataset.to_dataframe():
    #     print(d)
    #     break
    sw = process_ccd(train_inter)

    meta_information = sw.get_metadata()
    print(meta_information.columns)
    subjects = meta_information["subject"].unique()
    print(subjects)
    for i, x in enumerate(sw):
        print(x)
        print(len(x))
        if i == 5:
            break
