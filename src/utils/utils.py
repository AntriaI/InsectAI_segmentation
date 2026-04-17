#####################################################################################
## for configuration file handling
#####################################################################################

import json

class Configuration:
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.load_json()

    def load_json(self):
        try:
            with open(self.filepath, 'r') as json_file:
                data = json.load(json_file)
                for key, value in data.items():
                    setattr(self, key, value)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON: {e}")
        except FileNotFoundError:
            print("Error: Config file not found.")

    def get(self, key, default=None):
        return getattr(self, key, default)

#####################################################################################
## for logging
#####################################################################################
import logging
import os
from datetime import datetime


def logger(log_dir=None, log_filename="training.log", log_level=logging.INFO):
    """
    Sets up a logger for the project.

    Args:
        log_dir (str, optional): Directory where logs will be saved. If None, logs will not be saved to a file.
        log_filename (str): Name of the log file.
        log_level (int): Logging level. Defaults to logging.INFO.

    Returns:
        logger (logging.Logger): Configured logger instance.
    """
    #create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    #ensure no duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    #create console handler for printing logs to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists
        log_file_path = os.path.join(log_dir, log_filename)

        file_handler = logging.FileHandler(log_file_path, mode="a")
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    logger = logger(log_dir="experiments/exp_01/logs", log_filename="example.log")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")


