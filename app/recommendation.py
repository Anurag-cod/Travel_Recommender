# # travel_recommender/app/recommendation.py

# from sklearn.neighbors import NearestNeighbors
# from sklearn.decomposition import TruncatedSVD
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from app.logger import log_function_execution

# @log_function_execution
# def get_collab_recommendations(idx, knn, collab_features_reduced, k=5):
#     distances, indices = knn.kneighbors([collab_features_reduced[idx]], n_neighbors=k+1)
#     return indices.flatten()[1:]

# @log_function_execution
# def get_hybrid_recommendations(name, travel_data, cosine_sim, knn, collab_features_reduced, k=20):
#     if name not in travel_data['Name'].values:
#         raise ValueError("Destination not found in data.")
    
#     idx = travel_data[travel_data['Name'] == name].index[0]
#     content_indices = pd.Series(cosine_sim[idx]).nlargest(k).index
#     collab_indices = get_collab_recommendations(idx, knn, collab_features_reduced, k)
#     hybrid_indices = list(set(content_indices) | set(collab_indices))
#     hybrid_indices = hybrid_indices[:k]
#     return travel_data.iloc[hybrid_indices]
# travel_recommender/app/recommendation.py

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.logger import CustomLogger, create_log_path

# Set up logger
logger = CustomLogger(logger_name='recommendation_logger', log_filename=create_log_path('recommendation'))
logger.set_log_level()

def get_collab_recommendations(idx, knn, collab_features_reduced, k=5):
    """
    Generate collaborative recommendations based on nearest neighbors.

    Parameters:
    - idx (int): Index of the place.
    - knn (NearestNeighbors): Trained KNN model.
    - collab_features_reduced (ndarray): Feature matrix for collaborative filtering.
    - k (int): Number of recommendations to return.

    Returns:
    - ndarray: Indices of recommended places.
    """
    try:
        logger.save_logs(f"Generating collaborative recommendations for index {idx}", log_level='info')
        distances, indices = knn.kneighbors([collab_features_reduced[idx]], n_neighbors=k+1)
        logger.save_logs(f"Collaborative recommendations generated successfully for index {idx}", log_level='info')
        return indices.flatten()[1:]
    
    except Exception as e:
        logger.save_logs(f"Error generating collaborative recommendations: {str(e)}", log_level='error')
        raise


def get_hybrid_recommendations(name, travel_data, cosine_sim, knn, collab_features_reduced, k):
    """
    Generate hybrid recommendations using both content-based and collaborative filtering.

    Parameters:
    - name (str): Name of the destination.
    - travel_data (DataFrame): Travel data containing destination information.
    - cosine_sim (ndarray): Cosine similarity matrix for content-based filtering.
    - knn (NearestNeighbors): Trained KNN model for collaborative filtering.
    - collab_features_reduced (ndarray): Feature matrix for collaborative filtering.
    - k (int): Number of recommendations to return.

    Returns:
    - DataFrame: DataFrame containing the hybrid recommendations.
    """
    try:
        if name not in travel_data['Name'].values:
            raise ValueError("Destination not found in data.")
        
        logger.save_logs(f"Generating hybrid recommendations for destination: {name}", log_level='info')
        
        idx = travel_data[travel_data['Name'] == name].index[0]
        
        # Get content-based recommendations
        content_indices = pd.Series(cosine_sim[idx]).nlargest(k).index
        logger.save_logs(f"Content-based recommendations generated for {name}", log_level='info')
        
        # Get collaborative recommendations
        collab_indices = get_collab_recommendations(idx, knn, collab_features_reduced, k)
        
        # Merge both recommendations
        hybrid_indices = list(set(content_indices) | set(collab_indices))
        hybrid_indices = hybrid_indices[:k]
        
        logger.save_logs(f"Hybrid recommendations generated for {name}", log_level='info')
        return travel_data.iloc[hybrid_indices]
    
    except Exception as e:
        logger.save_logs(f"Error generating hybrid recommendations: {str(e)}", log_level='error')
        raise
