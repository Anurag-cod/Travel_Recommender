# # # travel_recommender/app.py
# import yaml
# from flask import Flask, render_template, request, jsonify
# from app.data_loader import load_travel_data
# from app.preprocess import preprocess_text_columns
# from app.recommendation import get_hybrid_recommendations
# from app.utils import get_unique_states
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# from sklearn.decomposition import TruncatedSVD
# from sklearn.neighbors import NearestNeighbors

# # Load configuration from config.yaml
# with open("config.yaml", "r") as config_file:
#     config = yaml.safe_load(config_file)

# # app = Flask(__name__)
# app = Flask(__name__)
# app.config['DEBUG'] = config['app']['debug']


# # Load and preprocess data
# # @log_function_execution
# def initialize_data():
#     travel_data = load_travel_data(config['data']['travel_data_file'])
#     text_columns = ['State', 'City', 'Type', 'Significance', 'Best Time to visit']
#     travel_data = preprocess_text_columns(travel_data, text_columns)
    
#     tfidf_vectorizer = TfidfVectorizer()
#     text_features = tfidf_vectorizer.fit_transform(travel_data[text_columns].apply(lambda x: ' '.join(x), axis=1))
#     cosine_sim = cosine_similarity(text_features, text_features)

#     collab_features = ['time needed to visit in hrs', 'Google review rating', 'Entrance Fee in INR', 'Number of google review in lakhs']
#     collab_data = travel_data[collab_features]
#     imputer = SimpleImputer(strategy='mean')
#     scaler = StandardScaler()
#     collab_preprocessor = make_pipeline(imputer, scaler)
#     collab_features_reduced = collab_preprocessor.fit_transform(collab_data)
#     svd = TruncatedSVD(n_components=4)
#     collab_pipeline = make_pipeline(collab_preprocessor, svd)
#     collab_features_reduced = collab_pipeline.fit_transform(collab_data)
#     knn = NearestNeighbors(metric='cosine', algorithm='brute')
#     knn.fit(collab_features_reduced)
    
#     return travel_data, cosine_sim, knn, collab_features_reduced

# travel_data, cosine_sim, knn, collab_features_reduced = initialize_data()

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     states = get_unique_states(travel_data)
#     recommendations = None
#     destinations = []
#     error_message = None
    
#     if request.method == 'POST':
#         state = request.form.get('state')
#         destination_name = request.form.get('destination')
        
#         if state:
#             state_data = travel_data[travel_data['State'].str.lower() == state.lower()]
#             if not state_data.empty:
#                 destinations = state_data['Name'].tolist()
                
#                 if destination_name:
#                     if destination_name in state_data['Name'].values:
#                         try:
#                             # recommendations = get_hybrid_recommendations(destination_name, travel_data, cosine_sim, knn, collab_features_reduced)
#                              recommendations = get_hybrid_recommendations(destination_name, travel_data,cosine_sim, knn, collab_features_reduced,k=config['recommender']['hybrid_recommendation_count'])
# #         return render_template('index.html', states=travel_data['State'].unique(), recommendations=recommendations)
#                         except ValueError as e:
#                             error_message = str(e)
#                     else:
#                         error_message = "Selected destination is not available in the state."
#                 else:
#                     error_message = "Please select a destination."
#             else:
#                 error_message = "No data found for the selected state."
            
#     return render_template('index.html', states=states, destinations=destinations, recommendations=recommendations, error_message=error_message)

# @app.route('/get_destinations', methods=['POST'])
# def get_destinations():
#     state = request.json.get('state')
#     if state:
#         state_data = travel_data[travel_data['State'].str.lower() == state.lower()]
#         if not state_data.empty:
#             destinations = state_data['Name'].tolist()
#         else:
#             destinations = []
#         return jsonify(destinations)
#     return jsonify([])

# if __name__ == '__main__':
#     app.run(host=config['app']['host'], port=config['app']['port'])

# travel_recommender/app.py
import yaml
import logging
from flask import Flask, render_template, request, jsonify
from app.data_loader import load_travel_data
from app.preprocess import preprocess_text_columns
from app.recommendation import get_hybrid_recommendations
from app.utils import get_unique_states
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from app.logger import CustomLogger, create_log_path

# Load configuration from config.yaml
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Initialize Flask app
app = Flask(__name__)
app.config['DEBUG'] = config['app']['debug']

# Set up logging
logger = CustomLogger(
    logger_name='app_logger',
    log_filename=create_log_path('app')
)
logger.set_log_level(logging.DEBUG)

# Load and preprocess data
def initialize_data():
    logger.save_logs("Initializing data loading and preprocessing.", log_level='info')
    
    try:
        travel_data = load_travel_data(config['data']['travel_data_file'])
        text_columns = ['State', 'City', 'Type', 'Significance', 'Best Time to visit']
        travel_data = preprocess_text_columns(travel_data, text_columns)
        
        tfidf_vectorizer = TfidfVectorizer()
        text_features = tfidf_vectorizer.fit_transform(travel_data[text_columns].apply(lambda x: ' '.join(x), axis=1))
        cosine_sim = cosine_similarity(text_features, text_features)

        collab_features = ['time needed to visit in hrs', 'Google review rating', 'Entrance Fee in INR', 'Number of google review in lakhs']
        collab_data = travel_data[collab_features]
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        collab_preprocessor = make_pipeline(imputer, scaler)
        collab_features_reduced = collab_preprocessor.fit_transform(collab_data)
        svd = TruncatedSVD(n_components=4)
        collab_pipeline = make_pipeline(collab_preprocessor, svd)
        collab_features_reduced = collab_pipeline.fit_transform(collab_data)
        knn = NearestNeighbors(metric='cosine', algorithm='brute')
        knn.fit(collab_features_reduced)
        
        logger.save_logs("Data initialization and preprocessing completed successfully.", log_level='info')
        return travel_data, cosine_sim, knn, collab_features_reduced

    except Exception as e:
        logger.save_logs(f"Error during data initialization: {e}", log_level='exception')
        raise

travel_data, cosine_sim, knn, collab_features_reduced = initialize_data()

@app.route('/', methods=['GET', 'POST'])
def index():
    states = get_unique_states(travel_data)
    recommendations = None
    destinations = []
    error_message = None
    
    if request.method == 'POST':
        state = request.form.get('state')
        destination_name = request.form.get('destination')
        
        if state:
            state_data = travel_data[travel_data['State'].str.lower() == state.lower()]
            if not state_data.empty:
                destinations = state_data['Name'].tolist()
                
                if destination_name:
                    if destination_name in state_data['Name'].values:
                        try:
                            # Get recommendations using hybrid method
                            recommendations = get_hybrid_recommendations(destination_name, travel_data, cosine_sim, knn, collab_features_reduced, k=config['recommender']['hybrid_recommendation_count'])
                            logger.save_logs(f"Recommendations successfully retrieved for destination: {destination_name}", log_level='info')
                        except ValueError as e:
                            error_message = str(e)
                            logger.save_logs(f"ValueError encountered: {e}", log_level='error')
                        except Exception as e:
                            error_message = 'An unexpected error occurred.'
                            logger.save_logs(f"Unexpected error: {e}", log_level='exception')
                    else:
                        error_message = "Selected destination is not available in the state."
                        logger.save_logs("Selected destination is not available in the state.", log_level='warning')
                else:
                    error_message = "Please select a destination."
                    logger.save_logs("Destination not provided by the user.", log_level='warning')
            else:
                error_message = "No data found for the selected state."
                logger.save_logs("No data found for the selected state.", log_level='warning')
            
    return render_template('index.html', states=states, destinations=destinations, recommendations=recommendations, error_message=error_message)

@app.route('/get_destinations', methods=['POST'])
def get_destinations():
    state = request.json.get('state')
    if state:
        state_data = travel_data[travel_data['State'].str.lower() == state.lower()]
        if not state_data.empty:
            destinations = state_data['Name'].tolist()
            logger.save_logs(f"Destinations successfully retrieved for state: {state}", log_level='info')
        else:
            destinations = []
            logger.save_logs(f"No destinations found for state: {state}", log_level='warning')
        return jsonify(destinations)
    return jsonify([])

if __name__ == '__main__':
    app.run(host=config['app']['host'], port=config['app']['port'])
