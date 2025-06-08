# Paths
RAW_DATA_PATH = 'data/raw_data.csv'
PROCESSED_DATA_PATH = 'data/processed_data.csv'
MODEL_PATH = 'model/spam_model.pkl'
VECTORIZER_PATH = 'model/tfidf_vectorizer.pkl'

# Model parameters
LOGISTIC_REGRESSION_PARAMS = {
    'penalty': 'l2',
    'C': 1.0,
    'solver': 'liblinear',
    'class_weight': 'balanced'
}

# TF-IDF parameters
TFIDF_PARAMS = {
    'max_features': 5000,
    'ngram_range': (1, 2)
}