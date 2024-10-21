import logging
import os

def setup_logging():
    """
        Setups the logging configuration for the project.

        Returns:
            logger: The logger object.
    """

    # Create logs directory if not exists
    logs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    # Creates the latest log file
    log_files = os.listdir(logs_path)
    log_files = [f for f in log_files if f.endswith(".log")]
    log_files.sort()
    if len(log_files) > 0:
        number = int(log_files[-1].split("_")[1].split(".")[0])
        number += 1
        log_path = os.path.join(logs_path, "train_{:04d}.log".format(number))
    else:
        log_path = os.path.join(logs_path, "train_0000.log")

    # Specifies the format of the log messages and returns the logger
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, stream_handler])

    return logging.getLogger(__name__)