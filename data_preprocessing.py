import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def prepare_data(data_path):
    """
    Prepare data for model training
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file containing the data
        
    Returns:
    --------
    tuple
        Processed features (X) and target variable (y)
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Encode categorical variables
    df = encode_categorical_variables(df)
    
    # Scale numerical features
    df = scale_numerical_features(df)
    
    # Separate features and target
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    return X, y

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    # Fill numerical missing values with median
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    
    # Fill categorical missing values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    return df

def encode_categorical_variables(df):
    """Encode categorical variables using Label Encoding"""
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    return df

def scale_numerical_features(df):
    """Scale numerical features using StandardScaler"""
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df 