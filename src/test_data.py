from data_utils import fixed_windows, PASSIVE_TASKS, ACTIVE_TASKS


r5 = fixed_windows(tasks="contrastChangeDetection", inter=[5])

meta_information = r5.get_metadata()
r5_subjects = meta_information["subject"].unique()


# from data_utils import EEGChallengeDataset, BaseConcatDataset, DATA_DIR, tqdm
# import numpy as np

inter = list(range(1, 12))
inter.remove(5)
print(inter)

for task in PASSIVE_TASKS + ACTIVE_TASKS:
    print(task)
    # dss = []
    # for i in tqdm(inter, desc="R"):
    #     try:
    #         dss.append(EEGChallengeDataset(
    #             task=task,
    #             release=f"R{i}",
    #             cache_dir=DATA_DIR,
    #             mini=False,
    #         ))
    #     except Exception as e:
    #         print(e)
    try:
        # concat_ds = BaseConcatDataset(dss)
        # T=np.array([ds.raw.n_times for ds in tqdm(concat_ds.datasets,desc="T")])
        # print(T.shape,T[T<200])
        rs = fixed_windows(tasks=task, inter=inter)
        meta_information = rs.get_metadata()
        rs_subjects = meta_information["subject"].unique()

        print(set(rs_subjects).intersection(r5_subjects))
    except Exception as e:
        print(e)
