import joblib
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

# Initialize text cleaning tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def load_models():
    """Load all the required models and encoders"""
    models = {
        'sentiment_model': joblib.load('models/sentiment_model.pkl'),
        'urgency_model': joblib.load('models/urgency_model.pkl'),
        'tfidf_vectorizer': joblib.load('models/tfidf_vectorizer.pkl'),
        'sentiment_encoder': joblib.load('models/sentiment_encoder.pkl'),
        'urgency_encoder': joblib.load('models/urgency_encoder.pkl')
    }
    return models

def clean_text(text):
    """Clean and preprocess the input text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Lemmatization and stopword removal
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

def predict_sentiment(text, models):
    """Predict sentiment from text"""
    cleaned_text = clean_text(text)
    text_vector = models['tfidf_vectorizer'].transform([cleaned_text])
    sentiment_pred = models['sentiment_model'].predict(text_vector)
    return models['sentiment_encoder'].inverse_transform(sentiment_pred)[0]

def predict_urgency(text, models):
    """Predict urgency from text"""
    cleaned_text = clean_text(text)
    text_vector = models['tfidf_vectorizer'].transform([cleaned_text])
    urgency_pred = models['urgency_model'].predict(text_vector)
    return models['urgency_encoder'].inverse_transform(urgency_pred)[0]

def get_emoji_mapping():
    """Return emoji mappings for sentiment and urgency"""
    return {
        'sentiment': {
            'Positive': '🟢',
            'Neutral': '🟡',
            'Negative': '🔴'
        },
        'urgency': {
            'Urgent': '⚠️',
            'Not Urgent': '✅'
        }
    }