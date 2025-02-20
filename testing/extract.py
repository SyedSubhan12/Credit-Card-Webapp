import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# Load the datasets
logging.info("Loading the datasets...")

df_1 = pd.read_csv("D:/Credit_Card_Spend_Analysis/data/credit_updated_1.csv")
logging.info(f"Dataset 1 loaded with {df_1.shape[0]} rows and {df_1.shape[1]} columns.")

df_2 = pd.read_csv("D:/Credit_Card_Spend_Analysis/data/credit_update_2.csv")
logging.info(f"Dataset 2 loaded with {df_2.shape[0]} rows and {df_2.shape[1]} columns.")

# Select relevant features from Dataset 1
df_1_selected = df_1[[
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "first", "last", 
    "gender", "street", "city", "state", "zip", "lat", "long", "city_pop", "job", "dob", 
    "trans_num", "unix_time", "merch_lat", "merch_long", "is_fraud", "merch_zipcode", 
    "Date", "Time", "company_parts", "merchant_fraud_rate"
]].copy()

# Select relevant features from Dataset 2
df_2_selected = df_2[[
    "Transaction ID", "Date", "Day of Week", "Time", "Type of Card", "Entry Mode", "Amount", 
    "Type of Transaction", "Merchant Group", "Country of Transaction", "Shipping Address", 
    "Country of Residence", "Gender", "Age", "Bank", "Fraud"
]].copy()
logging.info("Relevant features selected.")

# Convert merging columns to string type and strip whitespace
df_1_selected["cc_num"] = df_1_selected["cc_num"].astype(str).str.strip()
df_2_selected["Transaction ID"] = df_2_selected["Transaction ID"].astype(str).str.strip()
df_1_selected['trans_date_trans_time'] = pd.to_datetime(df_1_selected['trans_date_trans_time'])
df_1_selected['month'] = df_1_selected['trans_date_trans_time'].dt.month
# Check for partially matching transaction IDs using substring matching
df_2_selected["Transaction ID Truncated"] = df_2_selected["Transaction ID"].str[-4:]
df_1_selected["cc_num Truncated"] = df_1_selected["cc_num"].str[-4:]

common_keys = set(df_1_selected["cc_num Truncated"]) & set(df_2_selected["Transaction ID Truncated"])
logging.info(f"Common Transaction IDs (Last 4 Digits Matching): {len(common_keys)}")

# Debug merging process
df_debug = df_1_selected.merge(df_2_selected, left_on="cc_num Truncated", right_on="Transaction ID Truncated", how="outer", indicator=True)
logging.info(df_debug["_merge"].value_counts())

# Ensure at least some matching records exist before merging
if len(common_keys) > 0:
    logging.info("Merging the datasets on truncated transaction ID...")
    df_merged = df_1_selected.merge(df_2_selected, left_on="cc_num Truncated", right_on="Transaction ID Truncated", how="inner")
    
    # Remove duplicate and auxiliary columns after merging
    df_merged.drop(columns=[ "Transaction ID Truncated", "cc_num Truncated"], inplace=True)
    
    # Save the merged dataset
    output_path = "D:/Credit_Card_Spend_Analysis/data/merged_credit_data.csv"
    df_merged.to_csv(output_path, index=False)
    
    logging.info(f"Feature extraction and merging completed. Output saved as: {output_path}")
    print(f"Feature extraction and merging completed. Output saved as: {output_path}")
    
    # Display summary of merged data
    logging.info(f"Merged dataset contains {df_merged.shape[0]} rows and {df_merged.shape[1]} columns.")
else:
    logging.warning("No common Transaction IDs found, even with partial matching. Merging cannot be performed.")
