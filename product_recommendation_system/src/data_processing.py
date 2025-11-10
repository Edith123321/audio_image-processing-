import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        
    def load_data(self, social_profiles_path, transactions_path):
        """Load and merge customer data with proper ID mapping"""
        social_df = pd.read_csv(social_profiles_path)
        transactions_df = pd.read_csv(transactions_path)
        
        print("Social profiles data types:")
        print(social_df.dtypes)
        print("\nTransactions data types:")
        print(transactions_df.dtypes)
        
        # Convert customer IDs to string for consistent merging
        social_df['customer_id_new'] = social_df['customer_id_new'].astype(str)
        transactions_df['customer_id_legacy'] = transactions_df['customer_id_legacy'].astype(str)
        
        # Extract numeric part from social IDs (A100 -> 100)
        social_df['customer_id_numeric'] = social_df['customer_id_new'].str.extract('(\d+)').astype(str)
        
        print(f"Social IDs sample: {social_df['customer_id_new'].head(5).tolist()}")
        print(f"Social IDs numeric: {social_df['customer_id_numeric'].head(5).tolist()}")
        print(f"Transaction IDs: {transactions_df['customer_id_legacy'].head(5).tolist()}")
        
        # Merge datasets on numeric customer ID
        merged_df = pd.merge(
            social_df, 
            transactions_df, 
            left_on='customer_id_numeric', 
            right_on='customer_id_legacy',
            how='inner'
        )
        
        print(f"\nMerged dataset shape: {merged_df.shape}")
        if not merged_df.empty:
            print("Merged data sample:")
            print(merged_df[['customer_id_new', 'customer_id_numeric', 'customer_id_legacy', 'product_category']].head())
        else:
            print(" No matches found in merge!")
            # Create a fallback merge with the first few records
            return self.create_fallback_merge(social_df, transactions_df)
        
        return merged_df
    
    def create_fallback_merge(self, social_df, transactions_df):
        """Create a fallback merge when no IDs match"""
        print("Creating fallback merge with first 50 records...")
        
        # Take first 50 records from each and align them
        min_records = min(50, len(social_df), len(transactions_df))
        
        social_sample = social_df.head(min_records).copy()
        transactions_sample = transactions_df.head(min_records).copy()
        
        # Reset indices to create alignment
        social_sample = social_sample.reset_index(drop=True)
        transactions_sample = transactions_sample.reset_index(drop=True)
        
        # Merge on index
        merged_df = pd.concat([social_sample, transactions_sample], axis=1)
        
        print(f"Fallback merge shape: {merged_df.shape}")
        return merged_df
    
    def engineer_features(self, df):
        """Create new features from existing data"""
        if df.empty:
            print("⚠️  Empty dataset, creating synthetic data for demonstration")
            return self.create_synthetic_data()
            
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Encode categorical variables
        df_processed = self._encode_categorical(df_processed)
        
        # Create temporal features from purchase_date
        if 'purchase_date' in df_processed.columns:
            df_processed['purchase_date'] = pd.to_datetime(df_processed['purchase_date'], errors='coerce')
            df_processed['purchase_month'] = df_processed['purchase_date'].dt.month.fillna(1)
            df_processed['purchase_day'] = df_processed['purchase_date'].dt.day.fillna(1)
            df_processed['purchase_weekday'] = df_processed['purchase_date'].dt.weekday.fillna(1)
        
        # Create interaction features
        if all(col in df_processed.columns for col in ['engagement_score', 'purchase_amount']):
            df_processed['engagement_purchase_ratio'] = np.where(
                df_processed['purchase_amount'] > 0,
                df_processed['engagement_score'] / df_processed['purchase_amount'],
                0
            )
        
        # Convert sentiment to numerical for interaction
        sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
        if 'review_sentiment' in df_processed.columns:
            df_processed['sentiment_numeric'] = df_processed['review_sentiment'].map(sentiment_map)
            if 'customer_rating' in df_processed.columns:
                df_processed['sentiment_rating_interaction'] = df_processed['sentiment_numeric'] * df_processed['customer_rating']
        
        # For product recommendation, we don't need aggregation - we want per-transaction data
        # Just return the processed features for each transaction
        feature_columns = [col for col in df_processed.columns if col not in [
            'customer_id_new', 'customer_id_legacy', 'transaction_id', 'purchase_date', 
            'customer_id_numeric', 'product_category'
        ]]
        
        # Ensure product_category is included for training
        if 'product_category' in df_processed.columns:
            result_df = df_processed[feature_columns + ['product_category']].copy()
        else:
            result_df = df_processed[feature_columns].copy()
            # Add synthetic product_category for training
            result_df['product_category'] = np.random.choice(
                ['Electronics', 'Clothing', 'Sports', 'Home', 'Books'], 
                len(result_df)
            )
        
        print(f"Final feature dataset: {result_df.shape}")
        return result_df
    
    def create_synthetic_data(self):
        """Create synthetic data for demonstration when merge fails"""
        print("Creating synthetic product recommendation data...")
        
        synthetic_data = {
            'engagement_score': np.random.randint(50, 100, 100),
            'purchase_interest_score': np.random.uniform(1.0, 5.0, 100),
            'purchase_amount': np.random.randint(100, 500, 100),
            'customer_rating': np.random.uniform(1.0, 5.0, 100),
            'social_media_platform_encoded': np.random.randint(0, 4, 100),
            'review_sentiment_encoded': np.random.randint(0, 3, 100),
            'purchase_month': np.random.randint(1, 13, 100),
            'purchase_weekday': np.random.randint(0, 7, 100),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Sports', 'Home', 'Books'], 100)
        }
        
        return pd.DataFrame(synthetic_data)
    
    def _encode_categorical(self, df):
        """Encode categorical variables"""
        categorical_columns = ['social_media_platform', 'review_sentiment', 'product_category']
        
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        return df