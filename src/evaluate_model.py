#!/usr/bin/env python3
"""
Script to evaluate a trained sentiment analysis model on a test dataset
"""

import os
import sys
import argparse
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_curve, auc, confusion_matrix, classification_report
)

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import TextPreprocessor, load_dataset
from src.features import FeatureExtractor


def evaluate_model(args):
    """Evaluate a trained model on a test dataset"""
    print(f"Evaluating model {args.model_path} on dataset {args.dataset}...")
    
    # Step 1: Load model and feature extractor
    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)
        
    if args.feature_extractor_path is None:
        # Try to infer the path from the model path
        model_dir = os.path.dirname(args.model_path)
        feature_extractor_path = os.path.join(model_dir, 'feature_extractor.pkl')
    else:
        feature_extractor_path = args.feature_extractor_path
    
    with open(feature_extractor_path, 'rb') as f:
        feature_extractor = pickle.load(f)
    
    # Step 2: Load and preprocess the test dataset
    df = load_dataset(args.dataset)
    
    # Check that required columns exist in the dataframe
    if args.text_column not in df.columns:
        raise ValueError(f"Text column '{args.text_column}' not found in dataset")
    if args.target_column not in df.columns:
        raise ValueError(f"Target column '{args.target_column}' not found in dataset")
    
    preprocessor = TextPreprocessor(
        remove_stopwords=args.remove_stopwords,
        lemmatize=args.lemmatize
    )
    processed_df = preprocessor.preprocess_data(df, args.text_column, args.target_column)
    
    # Step 3: Extract features
    X = feature_extractor.transform(processed_df['cleaned_text'])
    y_true = processed_df[args.target_column]
    
    # Step 4: Make predictions
    y_pred = model.predict(X)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)
    else:
        y_proba = None
    
    # Step 5: Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # Step 6: Print results
    print("\nModel evaluation results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Step 7: Generate visualizations
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'confusion_matrix.png'))
    
    # ROC Curve (for binary classification)
    if y_proba is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(args.results_dir, 'roc_curve.png'))
    
    # Generate prediction errors file
    errors_df = pd.DataFrame({
        'text': df[args.text_column],
        'cleaned_text': processed_df['cleaned_text'],
        'true_label': y_true,
        'predicted_label': y_pred,
        'correct': y_true == y_pred
    })
    
    # Add probabilities if available
    if y_proba is not None:
        errors_df['confidence'] = np.max(y_proba, axis=1)
    
    # Save only the errors
    errors_df = errors_df[~errors_df['correct']]
    errors_df.to_csv(os.path.join(args.results_dir, 'prediction_errors.csv'), index=False)
    
    print(f"\nVisualizations and error analysis saved to {args.results_dir}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report,
        'confusion_matrix': cm
    }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate a sentiment analysis model')
    
    # Input arguments
    parser.add_argument('--model-path', type=str, default='models/sentiment_model.pkl',
                        help='Path to the saved model')
    parser.add_argument('--feature-extractor-path', type=str, default=None,
                        help='Path to the saved feature extractor')
    parser.add_argument('--dataset', type=str, default='data/imdb_sample.csv',
                        help='Path to the test dataset file')
    
    # Dataset arguments
    parser.add_argument('--text-column', type=str, default='review',
                        help='Name of the column containing text data')
    parser.add_argument('--target-column', type=str, default='sentiment',
                        help='Name of the column containing the target variable')
    
    # Preprocessing arguments
    parser.add_argument('--remove-stopwords', action='store_true',
                        help='Remove stopwords during preprocessing')
    parser.add_argument('--lemmatize', action='store_true',
                        help='Apply lemmatization during preprocessing')
    
    # Output arguments
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory to save evaluation results')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args)
