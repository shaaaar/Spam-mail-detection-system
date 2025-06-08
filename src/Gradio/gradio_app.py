import gradio as gr
import joblib
import re
import os

# Define paths
model_path = "C:/Users/shard/OneDrive/Documents/CODES/spam-detection-project/model/spam_model.pkl"
vectorizer_path = "C:/Users/shard/OneDrive/Documents/CODES/spam-detection-project/model/tfidf_vectorizer.pkl"

# Load model and vectorizer
try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model/vectorizer: {e}")

# Text cleaning function
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()

# Prediction function
def analyze_email(email_text):
    if not email_text.strip():
        return "⚠️ Please enter some email text."
    cleaned = clean_text(email_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return "🚫 SPAM" if prediction == 1 else "✅ NOT SPAM"

# Create Gradio interface
with gr.Blocks(title="Spam Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 📧 Spam Email Detector")
    gr.Markdown("Paste your email content below to check if it's spam or not.")
    
    email_input = gr.Textbox(label="Email Text", lines=15, placeholder="Enter your email here...")
    output = gr.Textbox(label="Prediction", interactive=False)

    analyze_btn = gr.Button("Analyze")
    analyze_btn.click(fn=analyze_email, inputs=email_input, outputs=output)

# Launch the app
if __name__ == "__main__":
    demo.launch()
