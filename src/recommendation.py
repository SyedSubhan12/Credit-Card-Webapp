import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

MODEL_FILE = "spending_forecast_model.pkl"
DATA_FILE = "D:/Credit_Card_Spend_Analysis/data/credit-2.csv"

def load_and_process_data():
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    df['Amount'] = df['Amount'].astype(str).str.replace('£', '', regex=True)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df['Month'] = df['Date'].dt.month
    
    # Aggregate monthly spending per user
    monthly_spending = df.groupby(['Shipping Address', 'Month'])['Amount'].sum().reset_index()
    return df, monthly_spending

def train_and_save_model(monthly_spending):
    if len(monthly_spending) < 2:
        print("Error: Not enough data for training. Ensure there are at least 2 valid monthly spending records.")
        return False
    
    X = monthly_spending[['Month']]
    y = monthly_spending['Amount']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    error = mean_absolute_error(y_test, y_pred)
    print(f"Model trained successfully. Mean Absolute Error: {error:.2f}")
    
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    return True

def load_model_and_predict():
    if not os.path.exists(MODEL_FILE):
        print("Model file not found. Training a new model...")
        df, monthly_spending = load_and_process_data()
        if not train_and_save_model(monthly_spending):
            print("Not enough data to proceed with training.")
            return
    
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    
    df, _ = load_and_process_data()
    transaction_id = input("Enter Transaction ID: ")
    user_info = df[df['Transaction ID'] == transaction_id]
    
    if user_info.empty:
        print("Transaction ID not found!")
        return
    
    user_id = user_info['Shipping Address'].values[0]
    user_month = user_info['Date'].dt.month.values[0]
    
    prediction = model.predict([[user_month]])[0]
    print(f"Predicted spending for user {user_id} in month {user_month}: {prediction:.2f}")

if __name__ == "__main__":
    load_model_and_predict()
