import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class EmailPreprocessor:
    def __init__(self):
        # Initialize NLTK components with error handling
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("NLTK resources missing. Downloading required data...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('punkt_tab')
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """Preprocess email text with robust error handling"""
        if not isinstance(text, str) or not text.strip():
            return ""
            
        try:
            text = text.lower()
            text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            text = re.sub(r'\S+@\S+', '', text)  # Remove emails
            text = re.sub(r'[^a-zA-Z\s]', '', text)  # Letters only
            tokens = nltk.word_tokenize(text)
            tokens = [word for word in tokens if word not in self.stop_words]
            tokens = [self.stemmer.stem(word) for word in tokens]
            return ' '.join(tokens)
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            return ""

    def read_text_files(self, folder_path, label):
        """Read text files with comprehensive error handling"""
        emails = []
        try:
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Directory not found: {folder_path}")
                
            for filename in os.listdir(folder_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()
                            cleaned = self.clean_text(text)
                            if cleaned:  # Only add if we got valid text
                                emails.append({'text': text, 'clean_text': cleaned, 'label': label})
                    except Exception as e:
                        print(f"Error reading {filename}: {str(e)}")
        except Exception as e:
            print(f"Error processing {folder_path}: {str(e)}")
        return emails

    def preprocess_text_files(self):
        """Main preprocessing function with path validation"""
        try:
            PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            spam_folder = os.path.join(PROJECT_DIR, 'data', 'raw', 'spam')
            ham_folder = os.path.join(PROJECT_DIR, 'data', 'raw', 'ham')
            output_path = os.path.join(PROJECT_DIR, 'data', 'processed_data.csv')
            
            print(f"Looking for spam emails in: {spam_folder}")
            print(f"Looking for ham emails in: {ham_folder}")
            
            spam_emails = self.read_text_files(spam_folder, 1)
            ham_emails = self.read_text_files(ham_folder, 0)
            
            if not spam_emails and not ham_emails:
                raise ValueError("No emails found in either folder. Please check your paths.")
            
            df = pd.DataFrame(spam_emails + ham_emails)
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            df.to_csv(output_path, index=False)
            print(f"Success! Processed data saved to {output_path}")
            print(f"Total emails processed: {len(df)} (Spam: {len(spam_emails)}, Ham: {len(ham_emails)})")
            return True
            
        except Exception as e:
            print(f"Fatal error in preprocessing: {str(e)}")
            return False

if __name__ == "__main__":
    print("Starting email preprocessing...")
    preprocessor = EmailPreprocessor()
    success = preprocessor.preprocess_text_files()
    if not success:
        print("Preprocessing failed. Please check the error messages above.")
        exit(1)