import mne
from eegdash.dataset import EEGChallengeDataset
from braindecode.preprocessing import create_windows_from_events,create_fixed_length_windows
from data_utils import PASSIVE_TASKS, ACTIVE_TASKS, SHIFT_AFTER_STIM, SFREQ, WINDOW_LEN, EPOCH_LEN_S

for task in PASSIVE_TASKS+ACTIVE_TASKS:
    print(task)
    dataset = EEGChallengeDataset(release="R5", cache_dir="data", task=task, mini=False)

    # raw = dataset.datasets[0].raw

    # events, events_ids = mne.events_from_annotations(raw)

    # print("ev",events)
    # print("ids",events_ids)
    print("================================================================")
    single_windows = create_fixed_length_windows( #create_windows_from_events(
        dataset,
        window_size_samples=int(EPOCH_LEN_S * SFREQ),
        window_stride_samples=SFREQ,
        preload=True,
        n_jobs=-1,
        drop_last_window=True,
    )

    for i,x in enumerate(single_windows):
        print(x)
        if i == 5:
            break

    break