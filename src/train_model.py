#!/usr/bin/env python3
"""
Script to train a sentiment analysis model and save it to disk
"""

import os
import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import TextPreprocessor, download_sample_dataset, load_dataset
from src.features import FeatureExtractor
from src.model import SentimentModel


def train_model(args):
    """Train a sentiment analysis model with the provided arguments"""
    print(f"Starting model training with {args.model_type} model...")
    
    # Step 1: Load or download dataset
    if not os.path.exists(args.dataset):
        print(f"Dataset {args.dataset} not found, downloading sample dataset...")
        df = download_sample_dataset(args.dataset, sample_size=args.sample_size)
    else:
        print(f"Loading dataset from {args.dataset}...")
        df = load_dataset(args.dataset)
        
    print(f"Dataset loaded with {len(df)} rows.")
    
    # Step 2: Preprocess the data
    print("Preprocessing text data...")
    preprocessor = TextPreprocessor(
        remove_stopwords=args.remove_stopwords,
        lemmatize=args.lemmatize
    )
    
    # Check that the text and target columns exist in the dataframe
    if args.text_column not in df.columns:
        raise ValueError(f"Text column '{args.text_column}' not found in dataset")
    if args.target_column not in df.columns:
        raise ValueError(f"Target column '{args.target_column}' not found in dataset")
    
    processed_df = preprocessor.preprocess_data(df, args.text_column, args.target_column)
    
    # Step 3: Split into train and test sets
    train_df, test_df = train_test_split(
        processed_df,
        test_size=args.test_size,
        random_state=42,
        stratify=processed_df[args.target_column]
    )
    
    print(f"Training set: {len(train_df)} rows")
    print(f"Test set: {len(test_df)} rows")
    
    # Step 4: Extract features
    print(f"Extracting features using {args.feature_method}...")
    feature_extractor = FeatureExtractor(
        method=args.feature_method,
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max)
    )
    
    X_train = feature_extractor.fit_transform(train_df['cleaned_text'])
    y_train = train_df[args.target_column]
    
    X_test = feature_extractor.transform(test_df['cleaned_text'])
    y_test = test_df[args.target_column]
    
    # Step 5: Train the model
    print(f"Training {args.model_type} model...")
    model = SentimentModel(
        model_type=args.model_type,
        class_weight=args.class_weight
    )
    
    if args.optimize:
        print("Optimizing hyperparameters (this may take a while)...")
        model.optimize_hyperparameters(X_train, y_train)
    else:
        model.train(X_train, y_train)
    
    # Step 6: Evaluate the model
    print("Evaluating model on test data...")
    results = model.evaluate(X_test, y_test)
    
    print("\nModel evaluation results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print("\nClassification Report:")
    print(results['report'])
    
    # Step 7: Save the model and feature extractor
    model_dir = os.path.dirname(args.model_path)
    os.makedirs(model_dir, exist_ok=True)
    
    model.save(args.model_path)
    
    feature_extractor_path = os.path.join(model_dir, "feature_extractor.pkl")
    import pickle
    with open(feature_extractor_path, "wb") as f:
        pickle.dump(feature_extractor, f)
    
    print(f"Model saved to {args.model_path}")
    print(f"Feature extractor saved to {feature_extractor_path}")
    
    # Step 8: Generate visualizations
    try:
        os.makedirs(args.results_dir, exist_ok=True)
        
        # Plot confusion matrix
        from sklearn.metrics import confusion_matrix
        import numpy as np
        
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(args.results_dir, 'confusion_matrix.png'))
        
        # If the model supports feature importance, plot it
        if hasattr(model.model, 'coef_'):
            # Get feature names
            feature_names = feature_extractor.get_feature_names()
            
            # For binary classification
            if len(model.model.coef_.shape) == 2:
                coefficients = model.model.coef_[0]
            else:
                coefficients = model.model.coef_
                
            # Get top features
            top_positive_idx = np.argsort(coefficients)[-15:]
            top_negative_idx = np.argsort(coefficients)[:15]
            
            # Combine indices and sort by absolute value
            top_idx = np.concatenate([top_positive_idx, top_negative_idx])
            top_coef = coefficients[top_idx]
            top_names = [feature_names[i] for i in top_idx]
            
            # Sort by coefficient value
            sorted_idx = np.argsort(top_coef)
            top_names = [top_names[i] for i in sorted_idx]
            top_coef = top_coef[sorted_idx]
            
            plt.figure(figsize=(10, 8))
            plt.barh(top_names, top_coef)
            plt.title('Top Features (Positive and Negative)')
            plt.xlabel('Coefficient')
            plt.tight_layout()
            plt.savefig(os.path.join(args.results_dir, 'feature_importance.png'))
            
        print(f"Visualizations saved to {args.results_dir}")
    except Exception as e:
        print(f"Warning: Failed to generate visualizations: {e}")
    
    return model, feature_extractor


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train a sentiment analysis model')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='data/imdb_sample.csv',
                        help='Path to the dataset file')
    parser.add_argument('--sample_size', type=int, default=200,
                        help='Number of samples to include in the sample dataset')
    parser.add_argument('--text-column', type=str, default='review',
                        help='Name of the column containing text data')
    parser.add_argument('--target-column', type=str, default='sentiment',
                        help='Name of the column containing the target variable')
    
    # Preprocessing arguments
    parser.add_argument('--remove-stopwords', action='store_true',
                        help='Remove stopwords during preprocessing')
    parser.add_argument('--lemmatize', action='store_true',
                        help='Apply lemmatization during preprocessing')
    
    # Feature extraction arguments
    parser.add_argument('--feature-method', type=str, default='tfidf',
                        choices=['bow', 'tfidf', 'tfidf-svd'],
                        help='Feature extraction method')
    parser.add_argument('--max-features', type=int, default=5000,
                        help='Maximum number of features to extract')
    parser.add_argument('--ngram-max', type=int, default=2,
                        help='Maximum n-gram size')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='logistic',
                        choices=['logistic', 'rf', 'svm', 'nb'],
                        help='Type of model to train')
    parser.add_argument('--class-weight', type=str, default='balanced',
                        help='Class weights for the model')
    parser.add_argument('--optimize', action='store_true',
                        help='Optimize hyperparameters using grid search')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    
    # Output arguments
    parser.add_argument('--model-path', type=str, default='models/sentiment_model.pkl',
                        help='Path to save the trained model')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory to save evaluation results')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
