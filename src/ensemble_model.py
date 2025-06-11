#!/usr/bin/env python3
"""
Enhanced ensemble sentiment analysis model with cross-validation and multiple techniques
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import TextPreprocessor, download_sample_dataset, load_dataset
from src.features import FeatureExtractor

class EnsembleSentimentModel:
    """
    Enhanced sentiment analysis model using ensemble methods and advanced techniques
    """
    
    def __init__(self):
        self.models = {}
        self.ensemble = None
        self.preprocessor = None
        self.feature_extractor = None
        self.is_trained = False
        
    def create_models(self):
        """Create individual models for the ensemble"""
        self.models = {
            'logistic': LogisticRegression(
                class_weight='balanced',
                C=1.0,
                solver='liblinear',
                random_state=42
            ),
            'rf': RandomForestClassifier(
                n_estimators=200,
                class_weight='balanced',
                random_state=42,
                max_depth=10
            ),
            'svm': SVC(
                class_weight='balanced',
                probability=True,
                random_state=42,
                kernel='linear'
            ),
            'nb': MultinomialNB(alpha=0.1)
        }
        
        # Create ensemble with voting
        self.ensemble = VotingClassifier(
            estimators=list(self.models.items()),
            voting='soft'  # Use probability-based voting
        )
        
    def train(self, X, y, feature_method='tfidf', max_features=5000, cv_folds=5):
        """
        Train the ensemble model with cross-validation
        """
        print("Creating preprocessing pipeline...")
        self.preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
        
        print("Extracting features...")
        self.feature_extractor = FeatureExtractor(
            method=feature_method,
            max_features=max_features,
            ngram_range=(1, 2)
        )
        
        # Fit feature extractor
        X_features = self.feature_extractor.fit_transform(X)
        
        print("Creating ensemble models...")
        self.create_models()
        
        print("Performing cross-validation...")
        cv_scores = {}
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            scores = cross_val_score(model, X_features, y, cv=skf, scoring='accuracy')
            cv_scores[name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            print(f"{name.capitalize()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        # Train ensemble
        print("Training ensemble model...")
        self.ensemble.fit(X_features, y)
        
        # Cross-validate ensemble
        ensemble_scores = cross_val_score(self.ensemble, X_features, y, cv=skf, scoring='accuracy')
        cv_scores['ensemble'] = {
            'mean': ensemble_scores.mean(),
            'std': ensemble_scores.std(),
            'scores': ensemble_scores
        }
        print(f"Ensemble: {ensemble_scores.mean():.4f} (+/- {ensemble_scores.std() * 2:.4f})")
        
        self.is_trained = True
        return cv_scores
    
    def predict(self, X):
        """Make predictions with the ensemble model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_features = self.feature_extractor.transform(X)
        return self.ensemble.predict(X_features)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_features = self.feature_extractor.transform(X)
        return self.ensemble.predict_proba(X_features)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        return metrics, y_pred
    
    def save_model(self, model_path):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'ensemble': self.ensemble,
            'preprocessor': self.preprocessor,
            'feature_extractor': self.feature_extractor
        }
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model_data, model_path)
        print(f"Enhanced model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load a trained model"""
        model_data = joblib.load(model_path)
        self.ensemble = model_data['ensemble']
        self.preprocessor = model_data['preprocessor']
        self.feature_extractor = model_data['feature_extractor']
        self.is_trained = True
        print(f"Enhanced model loaded from {model_path}")


def train_enhanced_model():
    """Train the enhanced ensemble model"""
    
    # Load or create dataset
    dataset_path = 'data/imdb_sample.csv'
    if not os.path.exists(dataset_path):
        print("Downloading sample dataset...")
        df = download_sample_dataset(dataset_path)
    else:
        print("Loading existing dataset...")
        df = load_dataset(dataset_path)
    
    print(f"Dataset loaded with {len(df)} rows")
    print(f"Class distribution: {df['sentiment'].value_counts().to_dict()}")
    
    # Preprocess data
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    processed_df = preprocessor.preprocess_data(df, 'review', 'sentiment')
    
    # Split data
    X = processed_df['cleaned_text']
    y = processed_df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create and train enhanced model
    model = EnsembleSentimentModel()
    cv_scores = model.train(X_train, y_train, feature_method='tfidf', max_features=8000)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics, y_pred = model.evaluate(X_test, y_test)
    
    print(f"\nTest Set Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the enhanced model
    model.save_model('models/enhanced_sentiment_model.pkl')
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot CV scores
    plt.subplot(2, 2, 1)
    model_names = list(cv_scores.keys())
    means = [cv_scores[name]['mean'] for name in model_names]
    stds = [cv_scores[name]['std'] for name in model_names]
    
    plt.bar(model_names, means, yerr=stds, capsize=5)
    plt.title('Cross-Validation Scores')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix
    plt.subplot(2, 2, 2)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Plot feature importance (for one of the models)
    plt.subplot(2, 2, 3)
    feature_names = model.feature_extractor.vectorizer.get_feature_names_out()
    if hasattr(model.models['logistic'], 'coef_'):
        coef = model.models['logistic'].coef_[0]
        top_indices = np.argsort(np.abs(coef))[-20:]
        top_features = [feature_names[i] for i in top_indices]
        top_coefs = coef[top_indices]
        
        plt.barh(range(len(top_features)), top_coefs)
        plt.yticks(range(len(top_features)), top_features)
        plt.title('Top Features (Logistic Regression)')
        plt.xlabel('Coefficient Value')
    
    plt.tight_layout()
    plt.savefig('results/enhanced_model_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved to results/enhanced_model_analysis.png")
    
    return model

if __name__ == "__main__":
    train_enhanced_model()
