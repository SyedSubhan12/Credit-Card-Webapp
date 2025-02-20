import pandas as pd
import numpy as np
import logging
from pandas.tseries.holiday import USFederalHolidayCalendar

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

class FeatureEngineering:
    """
    A class for performing feature engineering on a credit card transaction dataset.
    """
    def __init__(self, file_path):
        logging.info("Loading dataset...")
        try:
            self.df = pd.read_csv(file_path, parse_dates=['trans_date_trans_time'])
            self.df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)  # Drop unnecessary index column
            logging.info("Dataset loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise
    
    def add_temporal_features(self):
        """Adds temporal features to the dataset."""
        logging.info("Adding temporal features...")
        try:
            self.df['day_of_week'] = self.df['trans_date_trans_time'].dt.dayofweek
            self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
            self.df['month'] = self.df['trans_date_trans_time'].dt.month
            self.df['hour'] = self.df['trans_date_trans_time'].dt.hour
            self.df['quarter'] = self.df['trans_date_trans_time'].dt.quarter
            self.df['week_of_year'] = self.df['trans_date_trans_time'].dt.isocalendar().week
            self.df['year'] = self.df['trans_date_trans_time'].dt.year
            logging.info("Temporal features added successfully.")
        except Exception as e:
            logging.error(f"Error adding temporal features: {e}")
    
    def add_holiday_feature(self):
        """Adds a holiday flag based on US federal holidays."""
        logging.info("Adding holiday feature...")
        try:
            cal = USFederalHolidayCalendar()
            holidays = cal.holidays(start=self.df['trans_date_trans_time'].min(), 
                                    end=self.df['trans_date_trans_time'].max())
            self.df['is_holiday'] = self.df['trans_date_trans_time'].isin(holidays).astype(int)
            logging.info("Holiday feature added successfully.")
        except Exception as e:
            logging.error(f"Error adding holiday feature: {e}")
    
    def add_transaction_features(self):
        """Adds transaction-based features such as log-transformed amount."""
        logging.info("Adding transaction-based features...")
        try:
            self.df['log_amt'] = np.log1p(self.df['amt'])
            self.df['category_encoded'] = self.df['category'].astype('category').cat.codes
            self.df['merchant_encoded'] = self.df['merchant'].astype('category').cat.codes
            logging.info("Transaction features added successfully.")
        except Exception as e:
            logging.error(f"Error adding transaction features: {e}")
    
    def add_customer_features(self):
        """Adds customer demographic features such as age and gender encoding."""
        logging.info("Adding customer demographic features...")
        try:
            self.df['dob'] = pd.to_datetime(self.df['dob'], errors='coerce')
            self.df['age'] = pd.to_datetime('today').year - self.df['dob'].dt.year
            self.df['gender_encoded'] = self.df['gender'].map({'M': 0, 'F': 1})
            logging.info("Customer demographic features added successfully.")
        except Exception as e:
            logging.error(f"Error adding customer features: {e}")
    
    def add_location_features(self):
        """Adds location-based categorical encodings and distance calculation."""
        logging.info("Adding location-based features...")
        try:
            self.df['zip_encoded'] = self.df['zip'].astype('category').cat.codes
            self.df['state_encoded'] = self.df['state'].astype('category').cat.codes
            self.df['city_encoded'] = self.df['city'].astype('category').cat.codes
            self.df['distance_from_merchant'] = np.sqrt((self.df['lat'] - self.df['merch_lat'])**2 + (self.df['long'] - self.df['merch_long'])**2)
            logging.info("Location-based features added successfully.")
        except Exception as e:
            logging.error(f"Error adding location features: {e}")
    
    def add_fraud_related_features(self):
        """Adds features useful for fraud detection."""
        logging.info("Adding fraud-related features...")
        try:
            self.df['transaction_count'] = self.df.groupby('cc_num')['cc_num'].transform('count')
            self.df['avg_transaction_amt'] = self.df.groupby('cc_num')['amt'].transform('mean')
            logging.info("Fraud-related features added successfully.")
        except Exception as e:
            logging.error(f"Error adding fraud-related features: {e}")
    
    def process(self, output_path='processed_data.csv'):
        """Executes all feature engineering steps and saves the processed dataset."""
        logging.info("Starting feature engineering process...")
        try:
            self.add_temporal_features()
            self.add_holiday_feature()
            self.add_transaction_features()
            self.add_customer_features()
            self.add_location_features()
            self.add_fraud_related_features()
            self.df.to_csv(output_path, index=False)
            logging.info(f"Feature engineering completed and saved to '{output_path}'")
            print(f"Processed DataFrame shape: {self.df.shape}")
            print(self.df.head())
            return self.df
        except Exception as e:
            logging.error(f"Feature engineering process failed: {e}")
            raise

# Example usage
if __name__ == "__main__":
    fe = FeatureEngineering("D:/Credit_Card_Spend_Analysis/data/credit_card_transactions.csv")  # Update with your actual CSV file path
    processed_df = fe.process("feature_eng.csv")  # Update output path if needed
    print("Feature Engineering Completed. Processed data saved.")
