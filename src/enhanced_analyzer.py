#!/usr/bin/env python3
"""
Enhanced text analyzer using the ensemble model
"""

import os
import sys
import joblib
import argparse

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_text_enhanced(text, model_path='models/enhanced_sentiment_model.pkl'):
    """
    Analyze sentiment of text using the enhanced ensemble model
    """
    try:
        # Load the enhanced model
        model_data = joblib.load(model_path)
        ensemble = model_data['ensemble']
        preprocessor = model_data['preprocessor']
        feature_extractor = model_data['feature_extractor']
        
        # Preprocess the text
        cleaned_text = preprocessor.clean_text(text)
        
        # Extract features
        X_features = feature_extractor.transform([cleaned_text])
        
        # Make prediction
        prediction = ensemble.predict(X_features)[0]
        prediction_proba = ensemble.predict_proba(X_features)[0]
        
        # Get sentiment label
        sentiment = 'positive' if prediction == 1 else 'negative'
        confidence = max(prediction_proba) * 100
        
        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'negative': prediction_proba[0] * 100,
                'positive': prediction_proba[1] * 100
            }
        }
        
    except FileNotFoundError:
        print(f"Enhanced model not found at {model_path}")
        print("Please train the enhanced model first by running: python src/ensemble_model.py")
        return None
    except Exception as e:
        print(f"Error analyzing text: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Analyze text sentiment using enhanced ensemble model')
    parser.add_argument('text', nargs='?', help='Text to analyze')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--model-path', default='models/enhanced_sentiment_model.pkl', help='Path to the enhanced model')
    
    args = parser.parse_args()
    
    if args.interactive:
        print("=== Enhanced Sentiment Analysis Interactive Mode ===")
        print("Enter text to analyze sentiment. Type 'exit' or 'quit' to end.")
        print()
        
        while True:
            try:
                text = input("Enter text: ").strip()
                if text.lower() in ['exit', 'quit', '']:
                    break
                
                result = analyze_text_enhanced(text, args.model_path)
                if result:
                    print(f"\nAnalysis Result:")
                    print(f"Text: {result['text']}")
                    print(f"Cleaned Text: {result['cleaned_text']}")
                    print(f"Sentiment: {result['sentiment']}")
                    print(f"Confidence: {result['confidence']:.2f}%")
                    print(f"Probability Distribution:")
                    print(f"  negative: {result['probabilities']['negative']:.2f}%")
                    print(f"  positive: {result['probabilities']['positive']:.2f}%")
                    print()
                
            except KeyboardInterrupt:
                break
                
        print("Goodbye!")
        
    elif args.text:
        result = analyze_text_enhanced(args.text, args.model_path)
        if result:
            print(f"\nAnalysis Result:")
            print(f"Text: {result['text']}")
            print(f"Cleaned Text: {result['cleaned_text']}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            print(f"Probability Distribution:")
            print(f"  negative: {result['probabilities']['negative']:.2f}%")
            print(f"  positive: {result['probabilities']['positive']:.2f}%")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
