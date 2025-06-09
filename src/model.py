import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

class SentimentModel:
    """Class for training and using sentiment analysis models"""
    
    def __init__(self, model_type='logistic', class_weight='balanced'):
        """
        Initialize the model with configurable options
        
        Args:
            model_type (str): Type of model to use ('logistic', 'rf', 'svm', or 'nb')
            class_weight (str or dict): Class weights to use
        """
        self.model_type = model_type
        self.class_weight = class_weight
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the appropriate classifier model"""
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                C=1.0,
                class_weight=self.class_weight,
                solver='liblinear',
                max_iter=1000,
                random_state=42
            )
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                class_weight=self.class_weight,
                random_state=42
            )
        elif self.model_type == 'svm':
            self.model = LinearSVC(
                C=1.0,
                class_weight=self.class_weight,
                max_iter=10000,
                random_state=42
            )
        elif self.model_type == 'nb':
            # Naive Bayes doesn't support class_weight directly
            self.model = MultinomialNB(alpha=1.0)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """
        Train the model on the training data
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            self: Trained model
        """
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Features to predict on
            
        Returns:
            np.ndarray: Predicted labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get probability estimates for predictions
        
        Args:
            X: Features to predict on
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models like LinearSVC that don't have predict_proba
            decision = self.model.decision_function(X)
            return scipy.special.expit(decision)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'report': classification_report(y_test, y_pred)
        }
        
        return results
    
    def optimize_hyperparameters(self, X_train, y_train, param_grid=None):
        """
        Optimize hyperparameters using Grid Search
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid (dict, optional): Parameter grid to search
            
        Returns:
            self: Model with optimized hyperparameters
        """
        if param_grid is None:
            # Default parameter grids
            if self.model_type == 'logistic':
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['liblinear', 'lbfgs']
                }
            elif self.model_type == 'rf':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20]
                }
            elif self.model_type == 'svm':
                param_grid = {
                    'C': [0.1, 1.0, 10.0]
                }
            elif self.model_type == 'nb':
                param_grid = {
                    'alpha': [0.1, 0.5, 1.0, 2.0]
                }
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update model with the best estimator
        self.model = grid_search.best_estimator_
        
        return self
    
    def save(self, model_path):
        """
        Save the trained model to disk
        
        Args:
            model_path (str): Path to save the model
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, model_path):
        """
        Load a trained model from disk
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            SentimentModel: Loaded model
        """
        with open(model_path, 'rb') as f:
            return pickle.load(f)
