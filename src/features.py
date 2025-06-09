import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline


class FeatureExtractor:
    """Class for extracting features from text for sentiment analysis"""
    
    def __init__(self, method='tfidf', max_features=5000, ngram_range=(1, 2)):
        """
        Initialize the feature extractor with configurable options
        
        Args:
            method (str): Feature extraction method ('bow', 'tfidf', or 'tfidf-svd')
            max_features (int): Maximum features to extract
            ngram_range (tuple): Range of n-grams to consider (min_n, max_n)
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.extractor = None
        self._initialize_extractor()
    
    def _initialize_extractor(self):
        """Initialize the appropriate feature extraction method"""
        if self.method == 'bow':
            # Bag of Words
            self.extractor = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words='english'
            )
        elif self.method == 'tfidf':
            # TF-IDF
            self.extractor = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words='english'
            )
        elif self.method == 'tfidf-svd':
            # TF-IDF with Latent Semantic Analysis (SVD)
            tfidf = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words='english'
            )
            svd = TruncatedSVD(n_components=min(300, self.max_features))
            self.extractor = Pipeline([('tfidf', tfidf), ('svd', svd)])
        else:
            raise ValueError(f"Unknown feature extraction method: {self.method}")
    
    def fit_transform(self, texts):
        """
        Fit the extractor to the data and transform the texts to features
        
        Args:
            texts (list): List of text documents
            
        Returns:
            np.ndarray: Feature matrix
        """
        return self.extractor.fit_transform(texts)
    
    def transform(self, texts):
        """
        Transform texts to features using the fitted extractor
        
        Args:
            texts (list): List of text documents
            
        Returns:
            np.ndarray: Feature matrix
        """
        return self.extractor.transform(texts)
    
    def get_feature_names(self):
        """
        Get the names of the features extracted
        
        Returns:
            list: List of feature names
        """
        if self.method == 'tfidf-svd':
            # SVD components don't have interpretable feature names
            return [f"component_{i}" for i in range(self.extractor.named_steps['svd'].n_components)]
        else:
            return self.extractor.get_feature_names_out()


def extract_advanced_features(df, text_column):
    """
    Extract advanced linguistic features beyond bag-of-words
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the column containing text data
        
    Returns:
        pd.DataFrame: Dataframe with additional feature columns
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Text length
    result_df['text_length'] = df[text_column].apply(len)
    
    # Word count
    result_df['word_count'] = df[text_column].apply(lambda x: len(x.split()))
    
    # Average word length
    result_df['avg_word_length'] = df[text_column].apply(
        lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
    )
    
    # Exclamation mark count
    result_df['exclamation_count'] = df[text_column].apply(lambda x: x.count('!'))
    
    # Question mark count
    result_df['question_count'] = df[text_column].apply(lambda x: x.count('?'))
    
    # Capitalized word ratio
    result_df['capital_ratio'] = df[text_column].apply(
        lambda x: sum(1 for word in x.split() if word.isupper()) / len(x.split()) if len(x.split()) > 0 else 0
    )
    
    return result_df
