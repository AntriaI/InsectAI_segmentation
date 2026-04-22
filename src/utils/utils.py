#####################################################################################
## for configuration file handling
#####################################################################################

import json
import logging
import os


class Configuration:
    def __init__(self, filepath):
        self.filepath = filepath
        self.load_json()

    def load_json(self):
        try:
            with open(self.filepath, "r") as json_file:
                data = json.load(json_file)
                for key, value in data.items():
                    setattr(self, key, value)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON: {e}")
        except FileNotFoundError:
            print(f"Error: Config file not found at {self.filepath}")

    def get(self, key, default=None):
        return getattr(self, key, default)


#####################################################################################
## for logging
#####################################################################################

def logger(log_dir=None, log_filename="training.log", log_level=logging.INFO):
    """
    Sets up a logger for the project.

    Args:
        log_dir (str, optional): Directory where logs will be saved.
        log_filename (str): Name of the log file.
        log_level (int): Logging level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log = logging.getLogger(__name__)
    log.setLevel(log_level)
    log.propagate = False

    # Ensure no duplicate handlers
    if log.hasHandlers():
        log.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)

    # File handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, log_filename)

        file_handler = logging.FileHandler(log_file_path, mode="a")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)

    return log


if __name__ == "__main__":
    log = logger(log_dir="experiments/exp_01/logs", log_filename="example.log")
    log.info("This is an info message.")
    log.warning("This is a warning message.")
    log.error("This is an error message.")