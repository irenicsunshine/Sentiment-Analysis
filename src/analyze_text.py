#!/usr/bin/env python3
# Minor update: file touched for Git visibility
"""
Script to analyze sentiment of text using a trained model
"""

import os
import sys
import argparse
import pickle
import numpy as np

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import TextPreprocessor


def load_model_and_extractor(model_path, feature_extractor_path=None):
    """
    Load a trained model and feature extractor
    
    Args:
        model_path (str): Path to the saved model
        feature_extractor_path (str, optional): Path to the feature extractor
        
    Returns:
        tuple: (model, feature_extractor)
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    if feature_extractor_path is None:
        # Try to infer the path from the model path
        model_dir = os.path.dirname(model_path)
        feature_extractor_path = os.path.join(model_dir, 'feature_extractor.pkl')
    
    with open(feature_extractor_path, 'rb') as f:
        feature_extractor = pickle.load(f)
        
    return model, feature_extractor


def analyze_text(text, model_path, feature_extractor_path=None, output_probabilities=True):
    """
    Analyze the sentiment of text using a trained model
    
    Args:
        text (str): Text to analyze
        model_path (str): Path to the saved model
        feature_extractor_path (str, optional): Path to the feature extractor
        output_probabilities (bool): Whether to output prediction probabilities
        
    Returns:
        dict: Analysis results
    """
    # Load model and feature extractor
    model, feature_extractor = load_model_and_extractor(model_path, feature_extractor_path)
    
    # Preprocess the text
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    cleaned_text = preprocessor.clean_text(text)
    
    # Extract features
    features = feature_extractor.transform([cleaned_text])
    
    # Make prediction
    prediction = int(model.predict(features)[0])
    
    result = {
        'text': text,
        'cleaned_text': cleaned_text,
        'prediction': prediction,
        'sentiment': 'positive' if prediction == 1 else 'negative'
    }
    
    # Add probabilities if requested
    if output_probabilities and hasattr(model, 'predict_proba'):
        try:
            probabilities = model.predict_proba(features)[0]
            result['probabilities'] = {
                'negative': float(probabilities[0]),
                'positive': float(probabilities[1])
            }
            result['confidence'] = float(max(probabilities))
        except Exception as e:
            # Some models may not support probability prediction
            result['probabilities'] = None
            result['confidence'] = None
    
    return result


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze sentiment of text')
    
    parser.add_argument('text', nargs='?', help='Text to analyze')
    parser.add_argument('--model-path', type=str, default='models/sentiment_model.pkl',
                        help='Path to the saved model')
    parser.add_argument('--feature-extractor-path', type=str, default=None,
                        help='Path to the saved feature extractor')
    parser.add_argument('--no-proba', action='store_true',
                        help='Do not output prediction probabilities')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    
    return parser.parse_args()


def interactive_mode(model_path, feature_extractor_path, output_probabilities):
    """Run in interactive mode, allowing the user to input text repeatedly"""
    print("=== Sentiment Analysis Interactive Mode ===")
    print("Enter text to analyze sentiment. Type 'exit' or 'quit' to end.")
    print()
    
    while True:
        try:
            text = input("\nEnter text: ")
            if text.lower() in ('exit', 'quit', 'q'):
                break
                
            if not text.strip():
                continue
                
            result = analyze_text(text, model_path, feature_extractor_path, output_probabilities)
            
            print("\nAnalysis Result:")
            print(f"Sentiment: {result['sentiment'].upper()}")
            
            if 'confidence' in result and result['confidence'] is not None:
                print(f"Confidence: {result['confidence']:.2%}")
                
            if 'probabilities' in result and result['probabilities'] is not None:
                print(f"Probability Distribution:")
                for label, prob in result['probabilities'].items():
                    print(f"  {label}: {prob:.2%}")
                    
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function to run the script"""
    args = parse_args()
    
    if args.interactive:
        interactive_mode(args.model_path, args.feature_extractor_path, not args.no_proba)
        return
        
    if not args.text:
        print("Error: No text provided. Use --interactive for interactive mode.")
        return
        
    result = analyze_text(
        args.text, 
        args.model_path, 
        args.feature_extractor_path, 
        not args.no_proba
    )
    
    print("\nAnalysis Result:")
    print(f"Text: {result['text']}")
    print(f"Cleaned Text: {result['cleaned_text']}")
    print(f"Sentiment: {result['sentiment']}")
    
    if 'confidence' in result and result['confidence'] is not None:
        print(f"Confidence: {result['confidence']:.2%}")
        
    if 'probabilities' in result and result['probabilities'] is not None:
        print("Probability Distribution:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.2%}")


if __name__ == "__main__":
    main()
