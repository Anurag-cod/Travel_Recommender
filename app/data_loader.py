# travel_recommender/app/data_loader.py

# import pandas as pd
# # from app.logger import log_function_execution

# # @log_function_execution
# def load_travel_data(file_path):
#     return pd.read_csv(file_path)

import pandas as pd
from app.logger import CustomLogger, create_log_path

# Initialize the logger
logger = CustomLogger(logger_name='data_loader', log_filename=create_log_path('data_loader'))
logger.set_log_level()

def load_travel_data(file_path):
    """
    Loads the travel data from a CSV file and logs the execution.

    Parameters:
    - file_path (str): The path to the CSV file containing travel data.

    Returns:
    - pd.DataFrame: A pandas DataFrame with the travel data.
    """
    try:
        logger.save_logs(f"Starting to load travel data from {file_path}", log_level='info')
        data = pd.read_csv(file_path)
        logger.save_logs(f"Successfully loaded travel data from {file_path}", log_level='info')
        return data
    except Exception as e:
        logger.save_logs(f"Error loading travel data from {file_path}: {e}", log_level='error')
        raise e
