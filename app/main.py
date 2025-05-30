from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

app = Flask(__name__)

# Load models and encoders
try:
    sentiment_model = joblib.load('models/sentiment_model.pkl')
    urgency_model = joblib.load('models/urgency_model.pkl')
    tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    sentiment_encoder = joblib.load('models/sentiment_encoder.pkl')
    urgency_encoder = joblib.load('models/urgency_encoder.pkl')
except Exception as e:
    print(f"Error loading models: {e}")
    raise e

# Initialize text cleaning tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
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

def predict_email(text):
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Vectorize the text
    text_vector = tfidf_vectorizer.transform([cleaned_text])
    
    # Make predictions
    sentiment_pred = sentiment_model.predict(text_vector)
    urgency_pred = urgency_model.predict(text_vector)
    
    # Decode predictions
    sentiment_label = sentiment_encoder.inverse_transform(sentiment_pred)[0]
    urgency_label = urgency_encoder.inverse_transform(urgency_pred)[0]
    
    # Get prediction probabilities
    sentiment_probs = sentiment_model.predict_proba(text_vector)[0]
    urgency_probs = urgency_model.predict_proba(text_vector)[0]
    
    # Create probability mapping
    sentiment_prob_map = {
        label: round(prob * 100, 2)
        for label, prob in zip(sentiment_encoder.classes_, sentiment_probs)
    }
    
    urgency_prob_map = {
        label: round(prob * 100, 2)
        for label, prob in zip(urgency_encoder.classes_, urgency_probs)
    }
    
    return {
        'sentiment': sentiment_label,
        'urgency': urgency_label,
        'sentiment_probabilities': sentiment_prob_map,
        'urgency_probabilities': urgency_prob_map
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get text from form
        email_text = request.form['email_text']
        
        # Make prediction
        prediction = predict_email(email_text)
        
        # Map to emojis
        sentiment_emoji = {
            'Positive': '🟢',
            'Neutral': '🟡',
            'Negative': '🔴'
        }
        
        urgency_emoji = {
            'Urgent': '⚠️',
            'Not Urgent': '✅'
        }
        
        # Add emojis to response
        prediction['sentiment_emoji'] = sentiment_emoji.get(prediction['sentiment'], '')
        prediction['urgency_emoji'] = urgency_emoji.get(prediction['urgency'], '')
        
        return render_template('result.html', 
                             email_text=email_text,
                             prediction=prediction)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if request.method == 'POST':
        data = request.get_json()
        email_text = data.get('text', '')
        
        if not email_text:
            return jsonify({'error': 'No text provided'}), 400
            
        prediction = predict_email(email_text)
        
        return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)