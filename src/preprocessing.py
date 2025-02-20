import pandas as pd
import numpy as np
import logging
from scipy import stats
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreprocessor:
    def __init__(self, df):
        self.df = df.copy()
    
    def convert_to_datetime(self):
        """
        Converts 'trans_date_trans_time' column to datetime format.
        """
        logging.info("Converting transaction date to datetime...")
        self.df['ds'] = pd.to_datetime(self.df['trans_date_trans_time'])
    
    def select_relevant_features(self):
        """
        Selects relevant features for fraud detection and ensures 'is_fraud' is included.
        """
        logging.info("Selecting relevant features...")
        selected_columns = ['ds', 'is_fraud', 'amt', 'day_of_week', 'is_weekend', 
                            'month', 'quarter', 'week_of_year', 'year', 'log_amt', 
                            'distance_from_merchant']
        
        available_columns = [col for col in selected_columns if col in self.df.columns]
        self.df = self.df[available_columns]  # Keep only relevant features
        logging.info(f"Selected features: {self.df.columns.tolist()}")
    
    def handle_missing_values(self):
        """Handles missing values by imputing numerical features with the median."""
        logging.info("Handling missing values...")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if numeric_cols.empty:
            logging.warning("No numeric columns available for missing value imputation.")
            return
        
        num_imputer = SimpleImputer(strategy='median')
        self.df[numeric_cols] = num_imputer.fit_transform(self.df[numeric_cols])
        logging.info("Missing values handled successfully.")
    
    def remove_duplicates(self):
        """Removes duplicate rows from the dataset."""
        initial_shape = self.df.shape
        self.df.drop_duplicates(inplace=True)
        logging.info(f"Removed {initial_shape[0] - self.df.shape[0]} duplicate rows.")
    
    def handle_outliers(self):
        """Removes extreme outliers using Z-score filtering (excluding 'is_fraud')."""
        logging.info("Handling outliers...")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude 'is_fraud' from outlier detection
        if 'is_fraud' in numeric_cols:
            numeric_cols.remove('is_fraud')
        
        if not numeric_cols:
            logging.warning("No numeric columns found for outlier detection.")
            return
        
        z_scores = np.abs(stats.zscore(self.df[numeric_cols]))
        outlier_mask = (z_scores < 3).all(axis=1)
        
        remaining_rows = outlier_mask.sum()
        if remaining_rows > 0:
            self.df = self.df[outlier_mask].copy()
            logging.info(f"Outliers handled successfully. {remaining_rows} rows remain after filtering.")
        else:
            logging.warning("All rows removed due to outlier filtering. Retaining original data.")
    
    def preprocess(self):
        """Runs all preprocessing steps in order."""
        logging.info("Starting preprocessing...")
        
        self.convert_to_datetime()
        self.select_relevant_features()
        logging.info(f"Shape after feature selection: {self.df.shape}")
        
        self.handle_missing_values()
        logging.info(f"Shape after handling missing values: {self.df.shape}")
        
        self.remove_duplicates()
        logging.info(f"Shape after removing duplicates: {self.df.shape}")
        
        self.handle_outliers()
        logging.info(f"Shape after handling outliers: {self.df.shape}")
        
        logging.info("Preprocessing completed successfully.")
        return self.df

# Usage example
if __name__ == "__main__":
    # Make sure to use the correct path to your dataset
    df = pd.read_csv("D:/Credit_Card_Spend_Analysis/data/feature_eng.csv")
    preprocessor = DataPreprocessor(df)
    processed_df = preprocessor.preprocess()
    
    # Verify fraud labels after processing
    print("Fraud Label Distribution:")
    print(processed_df['is_fraud'].value_counts())
    
    processed_df.to_csv("processed_data.csv", index=False)
