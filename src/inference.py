import joblib
import re
import os

def load_email(file_path='src\email.txt'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    with open(file_path, 'r', encoding='utf-8') as file:
        email = file.read()
    return email

def clean_text(text):
    # Basic cleaning: remove URLs, special characters, digits
    text = re.sub(r'http\S+', '', text)            # remove URLs
    text = re.sub(r'\d+', '', text)                # remove digits
    text = re.sub(r'[^\w\s]', '', text)            # remove punctuation
    text = text.lower().strip()
    return text

def predict_email():
    # Load model and vectorizer
    model = joblib.load('model/spam_model.pkl')
    vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

    # Load and clean email
    raw_email = load_email()
    cleaned_email = clean_text(raw_email)

    # Transform and predict
    X = vectorizer.transform([cleaned_email])
    prediction = model.predict(X)[0]

    # Output result
    if prediction == 1:
        print("🚫 This email is **SPAM**.")
    else:
        print("✅ This email is **NOT SPAM**.")

if __name__ == "__main__":
    predict_email()
