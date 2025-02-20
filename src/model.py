import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
import pickle
import matplotlib.pyplot as plt

def load_and_preprocess_data(file_path):
    """
    Load the dataset, perform feature engineering, and create lag features.
    """
    df = pd.read_csv(file_path)
    df['ds'] = pd.to_datetime(df['ds'])
    df['hour'] = df['ds'].dt.hour
    df['minute'] = df['ds'].dt.minute
    df['second'] = df['ds'].dt.second
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['ds'].dt.month
    df['quarter'] = df['ds'].dt.quarter
    df['week_of_year'] = df['ds'].dt.isocalendar().week
    df['year'] = df['ds'].dt.year
    df = df.groupby('ds').mean().reset_index()
    
    # Creating lag and rolling features
    df['lag_1'] = df['amt'].shift(1)
    df['lag_7'] = df['amt'].shift(7)
    df['lag_30'] = df['amt'].shift(30)
    df['rolling_mean_7'] = df['amt'].rolling(7).mean()
    df['rolling_std_7'] = df['amt'].rolling(7).std()
    
    df.dropna(inplace=True)  # Drop rows with NaN values after feature creation
    
    features = ['hour', 'minute', 'second', 'day_of_week', 'is_weekend', 'month', 
                'quarter', 'week_of_year', 'year', 'lag_1', 'lag_7', 'lag_30', 
                'rolling_mean_7', 'rolling_std_7']
    X = df[features]
    y = df['amt']
    
    return df, X, y

def train_test_split_data(X, y):
    """Splits data into training and testing sets."""
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_xgboost():
    """Initializes the XGBoost model."""
    return XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        reg_lambda=2,
        reg_alpha=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

def train_model(model, X_train, y_train, X_test, y_test):
    """Trains the XGBoost model with early stopping."""
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model performance using various metrics."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")
    
    return y_pred

def plot_results(y_test, y_pred):
    """Plots actual vs predicted values."""
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    plt.show()

def save_model(model, filename="xgboost_model.pkl"):
    """Saves the trained model to a file."""
    with open(filename, "wb") as f:
        pickle.dump(model, f)

def load_model(filename="xgboost_model.pkl"):
    """Loads a saved model from a file."""
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_train_test_data(X_train, X_test, y_train, y_test, train_filename="train_data.csv", test_filename="test_data.csv"):
    """Saves the train and test data into CSV files."""
    train_data = X_train.copy()
    train_data['amt'] = y_train
    train_data.to_csv(train_filename, index=False)
    
    test_data = X_test.copy()
    test_data['amt'] = y_test
    test_data.to_csv(test_filename, index=False)

# Main execution (after splitting the data)
if __name__ == "__main__":
    file_path = "D:/Credit_Card_Spend_Analysis/data/processed_data.csv"  # Update with actual path
    df, X, y = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    
    # Save the training and testing data
    save_train_test_data(X_train, X_test, y_train, y_test)
    
    model = build_xgboost()
    model = train_model(model, X_train, y_train, X_test, y_test)
    
    y_pred = evaluate_model(model, X_test, y_test)
    plot_results(y_test, y_pred)
    
    save_model(model)  # Save trained model
