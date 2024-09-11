# # travel_recommender/app/preprocess.py

# import pandas as pd
# # from app.logger import log_function_execution

# # @log_function_execution
# def preprocess_text_columns(travel_data, text_columns):
#     def preprocess_text(text):
#         return ' '.join(str(text).lower().split())
    
#     for col in text_columns:
#         travel_data[col] = travel_data[col].apply(preprocess_text)
    
#     return travel_data
# travel_recommender/app/preprocess.py

import pandas as pd
from app.logger import CustomLogger, create_log_path

# Set up logger
logger = CustomLogger(logger_name='preprocess_logger', log_filename=create_log_path('preprocess'))
logger.set_log_level()

def preprocess_text_columns(travel_data, text_columns):
    """
    Preprocesses the text columns in the travel_data by converting to lowercase and removing extra spaces.

    Parameters:
    - travel_data (DataFrame): The travel data containing text columns.
    - text_columns (list): List of columns to preprocess.

    Returns:
    - DataFrame: The updated travel data with preprocessed text columns.
    """
    try:
        logger.save_logs(f"Preprocessing started for columns: {text_columns}", log_level='info')
        
        def preprocess_text(text):
            return ' '.join(str(text).lower().split())
        
        for col in text_columns:
            travel_data[col] = travel_data[col].apply(preprocess_text)
        
        logger.save_logs("Text columns preprocessed successfully", log_level='info')
        
        return travel_data
    
    except Exception as e:
        logger.save_logs(f"Error in preprocessing text columns: {str(e)}", log_level='error')
        raise
