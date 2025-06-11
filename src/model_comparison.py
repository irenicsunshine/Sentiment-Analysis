#!/usr/bin/env python3
"""
Comprehensive comparison of different sentiment analysis approaches
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import TextPreprocessor, load_dataset
from src.features import FeatureExtractor
from src.model import SentimentModel

def compare_models():
    """
    Compare different model configurations and feature extraction methods
    """
    print("=== Sentiment Analysis Model Comparison ===")
    
    # Load dataset
    dataset_path = 'data/imdb_sample.csv'
    df = load_dataset(dataset_path)
    print(f"Dataset loaded with {len(df)} rows")
    
    # Preprocess data
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    processed_df = preprocessor.preprocess_data(df, 'review', 'sentiment')
    
    X = processed_df['cleaned_text']
    y = processed_df['sentiment']
    
    # Define configurations to test
    configurations = [
        {
            'name': 'Logistic + TF-IDF',
            'model_type': 'logistic',
            'feature_method': 'tfidf',
            'max_features': 5000,
            'ngram_range': (1, 2)
        },
        {
            'name': 'Naive Bayes + TF-IDF',
            'model_type': 'nb',
            'feature_method': 'tfidf',
            'max_features': 5000,
            'ngram_range': (1, 2)
        },
        {
            'name': 'SVM + TF-IDF',
            'model_type': 'svm',
            'feature_method': 'tfidf',
            'max_features': 5000,
            'ngram_range': (1, 2)
        },
        {
            'name': 'Random Forest + TF-IDF',
            'model_type': 'rf',
            'feature_method': 'tfidf',
            'max_features': 5000,
            'ngram_range': (1, 2)
        },
        {
            'name': 'Logistic + BOW',
            'model_type': 'logistic',
            'feature_method': 'bow',
            'max_features': 5000,
            'ngram_range': (1, 2)
        },
        {
            'name': 'Naive Bayes + BOW',
            'model_type': 'nb',
            'feature_method': 'bow',
            'max_features': 5000,
            'ngram_range': (1, 2)
        }
    ]
    
    results = []
    
    # Test each configuration
    for config in configurations:
        print(f"\nTesting {config['name']}...")
        
        try:
            # Extract features
            feature_extractor = FeatureExtractor(
                method=config['feature_method'],
                max_features=config['max_features'],
                ngram_range=config['ngram_range']
            )
            X_features = feature_extractor.fit_transform(X)
            
            # Create model
            model = SentimentModel(config['model_type'])
            
            # Perform cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model.model, X_features, y, cv=cv, scoring='accuracy')
            
            result = {
                'Configuration': config['name'],
                'Model': config['model_type'],
                'Features': config['feature_method'],
                'CV_Mean': cv_scores.mean(),
                'CV_Std': cv_scores.std(),
                'CV_Min': cv_scores.min(),
                'CV_Max': cv_scores.max()
            }
            
            results.append(result)
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
        except Exception as e:
            print(f"Error with {config['name']}: {e}")
            continue
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('CV_Mean', ascending=False)
    
    print("\n=== RESULTS SUMMARY ===")
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    # Save results
    results_df.to_csv('results/model_comparison.csv', index=False)
    print(f"\nResults saved to results/model_comparison.csv")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Bar plot of CV scores
    plt.subplot(2, 2, 1)
    bars = plt.bar(range(len(results_df)), results_df['CV_Mean'], 
                   yerr=results_df['CV_Std'], capsize=5, 
                   color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray'][:len(results_df)])
    plt.title('Cross-Validation Accuracy Comparison')
    plt.xlabel('Model Configuration')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(results_df)), results_df['Configuration'], rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + results_df.iloc[i]['CV_Std'],
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Box plot showing score distribution
    plt.subplot(2, 2, 2)
    model_names = results_df['Configuration'].tolist()
    cv_scores_all = []
    
    for config in configurations:
        if config['name'] in model_names:
            try:
                feature_extractor = FeatureExtractor(
                    method=config['feature_method'],
                    max_features=config['max_features'],
                    ngram_range=config['ngram_range']
                )
                X_features = feature_extractor.fit_transform(X)
                model = SentimentModel(config['model_type'])
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model.model, X_features, y, cv=cv, scoring='accuracy')
                cv_scores_all.append(cv_scores)
            except:
                cv_scores_all.append([0])
    
    plt.boxplot(cv_scores_all, labels=[name.split(' + ')[0] for name in model_names])
    plt.title('Cross-Validation Score Distribution')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Feature method comparison
    plt.subplot(2, 2, 3)
    feature_comparison = results_df.groupby('Features')['CV_Mean'].mean().sort_values(ascending=False)
    bars = plt.bar(feature_comparison.index, feature_comparison.values, 
                   color=['skyblue', 'lightgreen'])
    plt.title('Feature Extraction Methods Comparison')
    plt.ylabel('Average CV Accuracy')
    plt.grid(True, alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Model type comparison
    plt.subplot(2, 2, 4)
    model_comparison = results_df.groupby('Model')['CV_Mean'].mean().sort_values(ascending=False)
    bars = plt.bar(model_comparison.index, model_comparison.values, 
                   color=['coral', 'lightblue', 'lightgreen', 'gold'])
    plt.title('Model Types Comparison')
    plt.ylabel('Average CV Accuracy')
    plt.grid(True, alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved to results/comprehensive_model_comparison.png")
    
    # Print recommendations
    print(f"\n=== RECOMMENDATIONS ===")
    best_config = results_df.iloc[0]
    print(f"Best Configuration: {best_config['Configuration']}")
    print(f"Accuracy: {best_config['CV_Mean']:.4f} (+/- {best_config['CV_Std'] * 2:.4f})")
    
    best_feature_method = feature_comparison.index[0]
    print(f"Best Feature Method: {best_feature_method}")
    
    best_model_type = model_comparison.index[0]
    print(f"Best Model Type: {best_model_type}")
    
    return results_df

if __name__ == "__main__":
    results = compare_models()
