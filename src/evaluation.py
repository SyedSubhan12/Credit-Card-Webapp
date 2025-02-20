import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

def load_model(model_path):
    """Load the trained XGBoost model."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_test_data(file_path):
    """Load test data with optimized memory usage."""
    dtype_dict = {'amt': 'float32'}  # Reduce float size to save memory
    test_data = pd.read_csv(file_path, dtype=dtype_dict)
    return test_data
def calculate_mape(y_true, y_pred):
    """Calculates the Mean Absolute Percentage Error, ignoring zero actual values."""
    non_zero_mask = y_true != 0
    y_true_non_zero = y_true[non_zero_mask]
    y_pred_non_zero = y_pred[non_zero_mask]
    
    if len(y_true_non_zero) == 0:
        return float('inf')  # Return inf if there are no non-zero values
    
    mape = np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100
    return mape

def make_predictions(model, X_test):
    """Make predictions using the trained XGBoost model."""
    predictions = model.predict(X_test)
    return predictions

def evaluate_model(y_test, y_pred):
    """Evaluate model performance using test data."""
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    non_zero_mask = y_test != 0
    y_true_non_zero = y_test[non_zero_mask]
    y_pred_non_zero = y_pred[non_zero_mask]

    if len(y_true_non_zero) == 0:
        mape = float('inf')  # Return inf if there are no non-zero values
    else:
        mape = np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100

    print("\nModel Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")

    
    
    return mae, mse, rmse, mape, y_test, y_pred
def plot_actual_vs_predicted(test_data, y_test, y_pred):
    """Plot actual vs predicted values to visually inspect discrepancies."""
    
    plt.figure(figsize=(14, 6))
    plt.plot(test_data['hour'], y_test, label='Actual', color="blue", alpha=0.7)
    plt.plot(test_data['hour'], y_pred, label="Predicted", color="red", linestyle="dashed", alpha=0.7)
    plt.xlabel("Hour")
    plt.ylabel("Transaction amount")
    plt.title("Actual vs Predicted transaction over time")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    model_path = "d:/Credit_Card_Spend_Analysis/models/xgboost_model.pkl"  # Update with actual path
    test_data_path = "d:/Credit_Card_Spend_Analysis/data/test_data.csv"    # Update with actual path

    model = load_model(model_path)
    test_data = load_test_data(test_data_path)
    
    features = ['hour', 'minute', 'second', 'day_of_week', 'is_weekend', 'month', 
                'quarter', 'week_of_year', 'year', 'lag_1', 'lag_7', 'lag_30', 
                'rolling_mean_7', 'rolling_std_7']
    
    X_test = test_data[features]
    y_test = test_data['amt']
    
    y_pred = make_predictions(model, X_test)
    
    # Correct function call
    evaluate_model(y_test, y_pred)
    
    # Correct function call
    plot_actual_vs_predicted(test_data, y_test, y_pred)
