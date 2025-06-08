import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import joblib
import os

def train_model():
    # Load processed data
    df = pd.read_csv('data/processed_data.csv')

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['clean_text'])
    y = df['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression with L2 regularization
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='liblinear',
        class_weight='balanced'
    )
    # # model.fit(X_train, y_train)

    # # Evaluate
    # y_pred = model.predict(X_test)
    # print(classification_report(y_test, y_pred))

    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)

    # Save model and vectorizer
    joblib.dump(model, 'model/spam_model.pkl')
    joblib.dump(tfidf, 'model/tfidf_vectorizer.pkl')
    print("Model saved successfully in model/ directory")

if __name__ == "__main__":
    train_model()
