#!/usr/bin/env python3
"""
Advanced error analysis tool for sentiment analysis model
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import TextPreprocessor, load_dataset
from src.features import FeatureExtractor
from src.model import SentimentModel
import joblib

class SentimentErrorAnalyzer:
    """
    Comprehensive error analysis for sentiment classification models
    """
    
    def __init__(self, model_path='models/sentiment_model.pkl', 
                 feature_extractor_path='models/feature_extractor.pkl'):
        self.model = joblib.load(model_path)
        
        self.feature_extractor = joblib.load(feature_extractor_path)
        self.preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
        
        self.errors_df = None
        self.correct_df = None
        
    def analyze_errors(self, dataset_path='data/imdb_sample.csv'):
        """
        Comprehensive error analysis on the dataset
        """
        print("=== Sentiment Analysis Error Analysis ===")
        
        # Load and preprocess data
        df = load_dataset(dataset_path)
        processed_df = self.preprocessor.preprocess_data(df, 'review', 'sentiment')
        
        X = processed_df['cleaned_text']
        y = processed_df['sentiment']
        
        # Make predictions
        X_features = self.feature_extractor.transform(X)
        predictions = self.model.predict(X_features)
        prediction_proba = self.model.predict_proba(X_features)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'text': df['review'],
            'cleaned_text': X,
            'true_label': y,
            'predicted_label': predictions,
            'correct': y == predictions,
            'confidence': np.max(prediction_proba, axis=1),
            'positive_prob': prediction_proba[:, 1],
            'negative_prob': prediction_proba[:, 0]
        })
        
        # Separate errors and correct predictions
        self.errors_df = results_df[~results_df['correct']].copy()
        self.correct_df = results_df[results_df['correct']].copy()
        
        print(f"Total samples: {len(results_df)}")
        print(f"Correct predictions: {len(self.correct_df)} ({len(self.correct_df)/len(results_df)*100:.1f}%)")
        print(f"Errors: {len(self.errors_df)} ({len(self.errors_df)/len(results_df)*100:.1f}%)")
        
        return results_df
    
    def analyze_error_patterns(self):
        """
        Analyze patterns in the errors
        """
        if self.errors_df is None:
            print("Please run analyze_errors() first")
            return
        
        print("\n=== Error Pattern Analysis ===")
        
        # Error types
        false_positives = self.errors_df[
            (self.errors_df['true_label'] == 0) & 
            (self.errors_df['predicted_label'] == 1)
        ]
        false_negatives = self.errors_df[
            (self.errors_df['true_label'] == 1) & 
            (self.errors_df['predicted_label'] == 0)
        ]
        
        print(f"False Positives (predicted positive, actually negative): {len(false_positives)}")
        print(f"False Negatives (predicted negative, actually positive): {len(false_negatives)}")
        
        # Confidence analysis for errors
        print(f"\nError Confidence Analysis:")
        print(f"Average confidence in errors: {self.errors_df['confidence'].mean():.3f}")
        print(f"Average confidence in correct predictions: {self.correct_df['confidence'].mean():.3f}")
        
        # Low confidence errors (uncertain predictions)
        low_conf_errors = self.errors_df[self.errors_df['confidence'] < 0.6]
        high_conf_errors = self.errors_df[self.errors_df['confidence'] >= 0.6]
        
        print(f"Low confidence errors (<60%): {len(low_conf_errors)}")
        print(f"High confidence errors (≥60%): {len(high_conf_errors)}")
        
        return {
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'low_conf_errors': low_conf_errors,
            'high_conf_errors': high_conf_errors
        }
    
    def analyze_text_features(self):
        """
        Analyze text features in errors
        """
        if self.errors_df is None:
            print("Please run analyze_errors() first")
            return
        
        print("\n=== Text Feature Analysis ===")
        
        # Text length analysis
        self.errors_df['text_length'] = self.errors_df['text'].str.len()
        self.errors_df['word_count'] = self.errors_df['text'].str.split().str.len()
        self.errors_df['cleaned_word_count'] = self.errors_df['cleaned_text'].str.split().str.len()
        
        print(f"Average text length in errors: {self.errors_df['text_length'].mean():.1f} characters")
        print(f"Average word count in errors: {self.errors_df['word_count'].mean():.1f} words")
        print(f"Average cleaned word count in errors: {self.errors_df['cleaned_word_count'].mean():.1f} words")
        
        # Compare with correct predictions
        if self.correct_df is not None:
            self.correct_df['text_length'] = self.correct_df['text'].str.len()
            self.correct_df['word_count'] = self.correct_df['text'].str.split().str.len()
            
            print(f"\nComparison with correct predictions:")
            print(f"Correct - Average text length: {self.correct_df['text_length'].mean():.1f} characters")
            print(f"Correct - Average word count: {self.correct_df['word_count'].mean():.1f} words")
    
    def find_problematic_words(self):
        """
        Find words that frequently appear in misclassified examples
        """
        if self.errors_df is None:
            print("Please run analyze_errors() first")
            return
        
        print("\n=== Problematic Words Analysis ===")
        
        # Extract words from error texts
        error_words = []
        for text in self.errors_df['cleaned_text']:
            error_words.extend(text.split())
        
        # Extract words from correct predictions
        correct_words = []
        for text in self.correct_df['cleaned_text']:
            correct_words.extend(text.split())
        
        # Count word frequencies
        error_word_counts = Counter(error_words)
        correct_word_counts = Counter(correct_words)
        
        # Find words that appear more frequently in errors
        problematic_words = []
        for word, error_count in error_word_counts.items():
            correct_count = correct_word_counts.get(word, 0)
            total_count = error_count + correct_count
            
            if total_count >= 2:  # Only consider words that appear at least twice
                error_rate = error_count / total_count
                if error_rate > 0.6:  # Words that lead to errors more than 60% of the time
                    problematic_words.append((word, error_rate, error_count, total_count))
        
        # Sort by error rate
        problematic_words.sort(key=lambda x: x[1], reverse=True)
        
        print("Top problematic words (high error rate):")
        for word, error_rate, error_count, total_count in problematic_words[:10]:
            print(f"  {word}: {error_rate:.2f} error rate ({error_count}/{total_count})")
        
        return problematic_words
    
    def create_visualizations(self, save_path='results/error_analysis.png'):
        """
        Create comprehensive error analysis visualizations
        """
        if self.errors_df is None:
            print("Please run analyze_errors() first")
            return
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Error distribution by confidence
        plt.subplot(2, 3, 1)
        bins = np.arange(0, 1.1, 0.1)
        plt.hist(self.errors_df['confidence'], bins=bins, alpha=0.7, color='red', label='Errors')
        plt.hist(self.correct_df['confidence'], bins=bins, alpha=0.7, color='green', label='Correct')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution: Errors vs Correct')
        plt.legend()
        
        # 2. Error types
        plt.subplot(2, 3, 2)
        error_types = ['False Positive', 'False Negative']
        false_pos = len(self.errors_df[(self.errors_df['true_label'] == 0) & 
                                      (self.errors_df['predicted_label'] == 1)])
        false_neg = len(self.errors_df[(self.errors_df['true_label'] == 1) & 
                                      (self.errors_df['predicted_label'] == 0)])
        
        plt.bar(error_types, [false_pos, false_neg], color=['lightcoral', 'lightblue'])
        plt.title('Error Types Distribution')
        plt.ylabel('Count')
        
        # 3. Text length vs confidence for errors
        plt.subplot(2, 3, 3)
        self.errors_df['text_length'] = self.errors_df['text'].str.len()
        plt.scatter(self.errors_df['text_length'], self.errors_df['confidence'], 
                   alpha=0.6, color='red')
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Confidence Score')
        plt.title('Text Length vs Confidence (Errors)')
        
        # 4. Word count distribution
        plt.subplot(2, 3, 4)
        self.errors_df['word_count'] = self.errors_df['text'].str.split().str.len()
        self.correct_df['word_count'] = self.correct_df['text'].str.split().str.len()
        
        plt.hist(self.errors_df['word_count'], bins=10, alpha=0.7, color='red', label='Errors')
        plt.hist(self.correct_df['word_count'], bins=10, alpha=0.7, color='green', label='Correct')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.title('Word Count Distribution')
        plt.legend()
        
        # 5. Prediction probability distribution for errors
        plt.subplot(2, 3, 5)
        plt.scatter(self.errors_df['positive_prob'], self.errors_df['negative_prob'], 
                   c=self.errors_df['true_label'], cmap='RdYlBu', alpha=0.7)
        plt.xlabel('Positive Probability')
        plt.ylabel('Negative Probability')
        plt.title('Probability Space (Errors)')
        plt.colorbar(label='True Label')
        
        # 6. Error rate by confidence bins
        plt.subplot(2, 3, 6)
        all_data = pd.concat([self.errors_df, self.correct_df])
        all_data['conf_bin'] = pd.cut(all_data['confidence'], bins=5)
        error_rate_by_conf = all_data.groupby('conf_bin')['correct'].apply(lambda x: 1 - x.mean())
        
        conf_labels = [f"{interval.left:.1f}-{interval.right:.1f}" for interval in error_rate_by_conf.index]
        plt.bar(range(len(error_rate_by_conf)), error_rate_by_conf.values)
        plt.xticks(range(len(error_rate_by_conf)), conf_labels, rotation=45)
        plt.xlabel('Confidence Bins')
        plt.ylabel('Error Rate')
        plt.title('Error Rate by Confidence Level')
        
        plt.tight_layout()
        
        # Save the plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error analysis visualization saved to {save_path}")
        
        return fig
    
    def generate_error_report(self, output_path='results/error_analysis_report.md'):
        """
        Generate a comprehensive error analysis report
        """
        if self.errors_df is None:
            print("Please run analyze_errors() first")
            return
        
        # Analyze error patterns
        patterns = self.analyze_error_patterns()
        problematic_words = self.find_problematic_words()
        
        # Generate report
        report = f"""# Sentiment Analysis Error Analysis Report

## Summary
- **Total Errors**: {len(self.errors_df)}
- **Error Rate**: {len(self.errors_df)/(len(self.errors_df)+len(self.correct_df))*100:.1f}%
- **False Positives**: {len(patterns['false_positives'])}
- **False Negatives**: {len(patterns['false_negatives'])}

## Confidence Analysis
- **Average Error Confidence**: {self.errors_df['confidence'].mean():.3f}
- **Average Correct Confidence**: {self.correct_df['confidence'].mean():.3f}
- **Low Confidence Errors (<60%)**: {len(patterns['low_conf_errors'])}
- **High Confidence Errors (≥60%)**: {len(patterns['high_conf_errors'])}

## Text Characteristics
- **Average Error Text Length**: {self.errors_df['text'].str.len().mean():.1f} characters
- **Average Error Word Count**: {self.errors_df['text'].str.split().str.len().mean():.1f} words

## Most Problematic Examples

### High Confidence Errors (Model was very wrong)
"""
        
        high_conf_errors = patterns['high_conf_errors'].sort_values('confidence', ascending=False)
        for idx, row in high_conf_errors.head(5).iterrows():
            sentiment = "Positive" if row['predicted_label'] == 1 else "Negative"
            true_sentiment = "Positive" if row['true_label'] == 1 else "Negative"
            report += f"- **Text**: \"{row['text'][:100]}...\"\n"
            report += f"  - **Predicted**: {sentiment} ({row['confidence']:.1%} confidence)\n"
            report += f"  - **Actual**: {true_sentiment}\n\n"
        
        report += """
## Problematic Words
Words that frequently appear in misclassified examples:
"""
        
        for word, error_rate, error_count, total_count in problematic_words[:10]:
            report += f"- **{word}**: {error_rate:.1%} error rate ({error_count}/{total_count} occurrences)\n"
        
        report += """
## Recommendations
1. **High Confidence Errors**: Review these examples as they indicate fundamental model misunderstandings
2. **Problematic Words**: Consider adding more training examples with these words in correct contexts
3. **Text Length**: Analyze if certain text lengths are systematically misclassified
4. **Feature Engineering**: Consider adding features that capture the problematic patterns identified

## Next Steps
1. Collect more training data addressing the identified error patterns
2. Experiment with different feature extraction methods
3. Consider ensemble methods to improve robustness
4. Implement confidence-based prediction thresholds
"""
        
        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Error analysis report saved to {output_path}")
        return report

def main():
    """
    Run comprehensive error analysis
    """
    analyzer = SentimentErrorAnalyzer()
    
    # Run analysis
    results_df = analyzer.analyze_errors()
    
    # Analyze patterns
    analyzer.analyze_text_features()
    analyzer.find_problematic_words()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Generate report
    analyzer.generate_error_report()
    
    print("\n=== Error Analysis Complete ===")
    print("Check the results/ folder for visualizations and detailed report.")

if __name__ == "__main__":
    main()
