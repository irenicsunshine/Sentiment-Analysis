#!/usr/bin/env python3
"""
Web application for sentiment analysis using Flask
"""

import os
import sys
import json
from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
import joblib

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import TextPreprocessor
from src.features import FeatureExtractor
from src.model import SentimentModel

app = Flask(__name__)

# Global variables for the model
model = None
feature_extractor = None
preprocessor = None
analysis_history = []

def load_model():
    """Load the trained sentiment analysis model"""
    global model, feature_extractor, preprocessor
    
    try:
        model = joblib.load('models/sentiment_model.pkl')
        feature_extractor = joblib.load('models/feature_extractor.pkl')
        preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def analyze_sentiment(text):
    """Analyze sentiment of given text"""
    if not model or not feature_extractor or not preprocessor:
        return None
    
    try:
        # Preprocess text
        cleaned_text = preprocessor.clean_text(text)
        
        # Extract features
        X_features = feature_extractor.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(X_features)[0]
        prediction_proba = model.predict_proba(X_features)[0]
        
        # Calculate sentiment score (-1 to 1)
        sentiment_score = (prediction_proba[1] - prediction_proba[0])
        
        # Determine sentiment intensity
        if sentiment_score > 0.6:
            intensity = "Very Positive"
        elif sentiment_score > 0.2:
            intensity = "Positive"
        elif sentiment_score > -0.2:
            intensity = "Neutral"
        elif sentiment_score > -0.6:
            intensity = "Negative"
        else:
            intensity = "Very Negative"
        
        result = {
            'text': text,
            'cleaned_text': cleaned_text,
            'sentiment': 'positive' if prediction == 1 else 'negative',
            'sentiment_score': sentiment_score,
            'intensity': intensity,
            'confidence': max(prediction_proba) * 100,
            'probabilities': {
                'negative': prediction_proba[0] * 100,
                'positive': prediction_proba[1] * 100
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to history
        analysis_history.append(result)
        
        return result
        
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return None

def create_sentiment_chart():
    """Create a sentiment distribution chart from history"""
    if not analysis_history:
        return None
    
    try:
        df = pd.DataFrame(analysis_history)
        
        plt.figure(figsize=(10, 6))
        
        # Sentiment distribution
        plt.subplot(1, 2, 1)
        sentiment_counts = df['sentiment'].value_counts()
        colors = ['lightcoral' if x == 'negative' else 'lightblue' for x in sentiment_counts.index]
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
        plt.title('Sentiment Distribution')
        
        # Sentiment score distribution
        plt.subplot(1, 2, 2)
        plt.hist(df['sentiment_score'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Sentiment Score')
        plt.ylabel('Frequency')
        plt.title('Sentiment Score Distribution')
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url
        
    except Exception as e:
        print(f"Error creating chart: {e}")
        return None

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze sentiment endpoint"""
    data = request.get_json()
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    result = analyze_sentiment(text)
    
    if result:
        return jsonify(result)
    else:
        return jsonify({'error': 'Failed to analyze sentiment'}), 500

@app.route('/history')
def history():
    """Get analysis history"""
    return jsonify(analysis_history)

@app.route('/stats')
def stats():
    """Get statistics and visualizations"""
    chart_url = create_sentiment_chart()
    
    stats = {
        'total_analyses': len(analysis_history),
        'positive_count': sum(1 for a in analysis_history if a['sentiment'] == 'positive'),
        'negative_count': sum(1 for a in analysis_history if a['sentiment'] == 'negative'),
        'average_confidence': sum(a['confidence'] for a in analysis_history) / len(analysis_history) if analysis_history else 0,
        'chart_url': chart_url
    }
    
    return jsonify(stats)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear analysis history"""
    global analysis_history
    analysis_history = []
    return jsonify({'message': 'History cleared'})

if __name__ == '__main__':
    print("Loading sentiment analysis model...")
    if load_model():
        print("Model loaded successfully!")
        print("Starting web application...")
        print("Access the application at: http://localhost:5001")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("Failed to load model. Please ensure the model files exist.")
        print("Run 'python src/train_model.py' to train a model first.")
