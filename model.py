import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ChurnPredictor:
    def __init__(self, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state
        )
        
    def train(self, X, y):
        """
        Train the model on the given data
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Calculate and store metrics
        y_pred = self.model.predict(X_val)
        self.metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred)
        }
        
        return self.metrics
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        array-like of shape (n_samples,)
            Predicted class labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        array-like of shape (n_samples, n_classes)
            Probability of each class for each sample
        """
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """
        Get the importance of each feature in the model
        
        Returns:
        --------
        dict
            Dictionary mapping feature names to their importance scores
        """
        return dict(zip(
            [f"feature_{i}" for i in range(len(self.model.feature_importances_))],
            self.model.feature_importances_
        )) 