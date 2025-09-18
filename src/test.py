import mne
from eegdash.dataset import EEGChallengeDataset
from braindecode.preprocessing import (
    create_windows_from_events,
    create_fixed_length_windows,
)
from data_utils import (
    PASSIVE_TASKS,
    ACTIVE_TASKS,
    SHIFT_AFTER_STIM,
    SFREQ,
    WINDOW_LEN,
    EPOCH_LEN_S,
    process_ccd,
    DATA_DIR,
)

for task in ["contrastChangeDetection"]:  # PASSIVE_TASKS+ACTIVE_TASKS:
    print(task)
    dataset = EEGChallengeDataset(
        release="R5", cache_dir=DATA_DIR, task=task, mini=False
    )

    # raw = dataset.datasets[0].raw

    # events, events_ids = mne.events_from_annotations(raw)

    # print("ev",events)
    # print("ids",events_ids)
    print("================================================================")
    single_windows = create_fixed_length_windows(  # create_windows_from_events(
        dataset,
        # start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),  # +0.5 s
        # stop_offset_samples=int(
        #     (SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ
        # ),  # +2.5 s
        window_size_samples=int(EPOCH_LEN_S * SFREQ),
        window_stride_samples=SFREQ,
        preload=True,
        n_jobs=-1,
        drop_last_window=True,
    )
    print(len(single_windows))
    # sw = process_ccd([5])
    # print(len(sw))
    # for x,y in zip(sw,single_windows):
    #     print(x)
    #     print(y)
    #     print(x[0]==y[0])
    #     break
