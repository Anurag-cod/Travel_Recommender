import logging
from pathlib import Path
import datetime as dt

def create_log_path(module_name: str) -> str:
    """
    Create a log file path based on the current date and the provided module_name.
    """
    current_date = dt.date.today()
    # Create a logs folder in the root directory
    root_path = Path(__file__).parent.parent
    log_dir_path = root_path / 'logs'
    log_dir_path.mkdir(exist_ok=True)
    
    # Create folder for a specific module
    module_log_path = log_dir_path / module_name
    module_log_path.mkdir(exist_ok=True, parents=True)
    
    # Create log file based on the current date
    current_date_str = current_date.strftime("%d-%m-%Y")
    log_file_name = module_log_path / (current_date_str + '.log')
    return log_file_name


class CustomLogger:
    def __init__(self, logger_name, log_filename):
        """
        Initializes a custom logger with the specified name and log file.
        """
        self.__logger = logging.getLogger(logger_name)
        self.__log_path = log_filename
        
        # Check if the logger already has handlers to prevent duplicates
        if not self.__logger.hasHandlers():
            file_handler = logging.FileHandler(filename=self.__log_path, mode='a')
            log_format = "%(asctime)s - %(levelname)s : %(message)s"
            time_format = '%d-%m-%Y %H:%M:%S'
            formatter = logging.Formatter(fmt=log_format, datefmt=time_format)
            file_handler.setFormatter(formatter)
            self.__logger.addHandler(file_handler)

    def get_log_path(self):
        """
        Returns the path to the log file.
        """
        return self.__log_path

    def get_logger(self):
        """
        Returns the logger object.
        """
        return self.__logger

    def set_log_level(self, level=logging.DEBUG):
        """
        Sets the log level for the logger.
        """
        self.__logger.setLevel(level)

    def save_logs(self, msg, log_level='info'):
        """
        Saves logs to the specified log file with the given message and log level.
        """
        if log_level == 'debug':
            self.__logger.debug(msg)
        elif log_level == 'info':
            self.__logger.info(msg)
        elif log_level == 'warning':
            self.__logger.warning(msg)
        elif log_level == 'error':
            self.__logger.error(msg)
        elif log_level == 'exception':
            self.__logger.exception(msg)
        elif log_level == 'critical':
            self.__logger.critical(msg)


if __name__ == "__main__":
    logger = CustomLogger(logger_name='my_logger', log_filename=create_log_path('test'))
    logger.set_log_level()
    logger.save_logs('save me, code is breaking', log_level='critical')
