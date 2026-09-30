"""Check how much subject overlap exists between release R5 and the other releases."""

import argparse
import logging

from data_utils import ACTIVE_TASKS, PASSIVE_TASKS, fixed_windows

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main(cache_dir: str) -> None:
    r5 = fixed_windows(tasks="contrastChangeDetection", inter=[5], cache_dir=cache_dir)
    r5_subjects = r5.get_metadata()["subject"].unique()

    inter = list(range(1, 12))
    inter.remove(5)
    logging.info("Releases: %s", inter)

    for task in PASSIVE_TASKS + ACTIVE_TASKS:
        logging.info("task %s", task)
        try:
            rs = fixed_windows(tasks=task, inter=inter, cache_dir=cache_dir)
            rs_subjects = rs.get_metadata()["subject"].unique()
            logging.info(
                "subject intersection %s", set(rs_subjects).intersection(r5_subjects)
            )
        except Exception:
            logging.exception("Failed to build fixed windows for task %s", task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default="./data",
        help="Directory holding the cached EEG dataset.",
    )
    main(parser.parse_args().cache_dir)
