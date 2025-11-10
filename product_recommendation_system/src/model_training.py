import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def prepare_facial_data(self, image_features_df):
        """Prepare data for facial recognition model"""
        X = image_features_df.drop(['member_id', 'image_file', 'augmentation', 'expression'], axis=1)
        y = image_features_df['member_id']
        
        # Handle missing values
        X = X.fillna(0)
        
        return X, y
    
    def prepare_audio_data(self, audio_features_df):
        """Prepare data for voice verification model"""
        X = audio_features_df.drop(['member_id', 'audio_file', 'augmentation', 'phrase'], axis=1)
        y = audio_features_df['member_id']
        
        # Handle missing values
        X = X.fillna(0)
        
        return X, y
    
    def prepare_product_data(self, merged_df):
        """Prepare data for product recommendation model"""
        X = merged_df.drop(['product_category', 'customer_id_new', 'customer_id_legacy', 
                           'transaction_id', 'purchase_date'], axis=1, errors='ignore')
        y = merged_df['product_category']
        
        # Handle categorical variables and missing values
        X = pd.get_dummies(X)
        X = X.fillna(0)
        
        return X, y
    
    def train_model(self, X, y, model_type='random_forest', model_name='default'):
        """Train a model with given parameters"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[model_name] = scaler
        
        # Initialize model
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(random_state=42)
        elif model_type == 'xgboost':
            model = XGBClassifier(random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"{model_name} - {model_type} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(classification_report(y_test, y_pred))
        
        self.models[model_name] = model
        return model, accuracy, f1
    
    def save_models(self, models_dir):
        """Save trained models and scalers"""
        for name, model in self.models.items():
            joblib.dump(model, f"{models_dir}/{name}_model.pkl")
        
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{models_dir}/{name}_scaler.pkl")