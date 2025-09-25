from data_utils import fixed_windows, PASSIVE_TASKS, ACTIVE_TASKS


r5 = fixed_windows(tasks="contrastChangeDetection", inter=[5])

meta_information = r5.get_metadata()
r5_subjects = meta_information["subject"].unique()

inter = list(range(1, 12))
inter.remove(5)
print("R", inter)

for task in PASSIVE_TASKS + ACTIVE_TASKS:
    print("task", task)
    try:
        rs = fixed_windows(tasks=task, inter=inter, cache_dir="/home/a/startkit/data")
        meta_information = rs.get_metadata()
        rs_subjects = meta_information["subject"].unique()

        print("subject intersection", set(rs_subjects).intersection(r5_subjects))
    except Exception as e:
        print(e)
