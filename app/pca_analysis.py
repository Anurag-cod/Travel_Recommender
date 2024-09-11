# travel_recommender/app/pca_analysis.py

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from app.logger import log_function_execution

@log_function_execution
def perform_pca(data_scaled):
    pca = PCA()
    pca.fit(data_scaled)
    explained_variance = pca.explained_variance_ratio_

    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.title('PCA Explained Variance')
    plt.show()
