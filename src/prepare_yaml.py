import logging
from os import getenv

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def prepare_config() -> None:
    """
    Set accelerate config based on environment variables.
    """

    base_config = getenv("CONFIG_FILE")
    if base_config is None:
        logging.error("There is no CONFIG_FILE")
        return
    logging.info(f"Load CONFIG_FILE from {base_config}")
    with open(base_config, encoding="utf-8") as file:
        accelerate_config: dict = yaml.safe_load(file)

    rank = getenv("RANK")
    if rank is None:
        logging.warning("There is no rank. Default set to 0")
    else:
        rank = int(rank)
        logging.info(f"The rank is {rank}")
        if accelerate_config.get("machine_rank") is not None:
            logging.info("Update rank in config")
            accelerate_config["machine_rank"] = rank

    host = getenv("MASTER_ADDR")
    if host is None:
        logging.warning('There is no "MASTER_ADDR".')
    else:
        logging.info(f"The 'MASTER_ADDR' is {host}")
        if accelerate_config.get("main_process_ip") is not None:
            logging.info("Update ip in config")
            accelerate_config["main_process_ip"] = host

    port = getenv("MASTER_PORT")
    if port is None:
        logging.warning("There is no 'MASTER_PORT'.")
    else:
        logging.info(f"The port: {port}")

        if accelerate_config.get("main_process_port") is not None:
            logging.info("Update port in config")
            accelerate_config["main_process_port"] = int(port)

    logging.info(f"The final accelerate config: {accelerate_config}")
    logging.info("Save the local accelerate config to 'acc_cfg.yml'.")
    with open("acc_cfg.yml", "w", encoding="utf-8") as file:
        yaml.safe_dump(accelerate_config, file, default_flow_style=False)


if __name__ == "__main__":
    prepare_config()
