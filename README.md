# Credit Card Spending Forecast Dashboard

## 📌 Overview
The **Credit Card Spending Forecast Dashboard** is a **Streamlit-powered web application** designed to analyze and forecast credit card spending trends using **interactive visualizations and machine learning models**. It provides insights into historical spending patterns and predicts future spending behavior using an **XGBoost-based predictive model**.

## 🚀 Features
- **📊 Interactive Data Visualization**: View historical spending trends via dynamic charts.
- **🔮 Future Forecasting**: Predict spending trends for the next 7-180 days using an XGBoost model.
- **🔎 User Input Panel**: Select customer ID, spending category, and time range.
- **📈 Fraud Detection Insights**: Identify potential fraudulent transactions.
- **💳 Payment Analysis**: Understand spending habits based on transaction methods.

## 🛠️ Installation Guide
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip
- Streamlit
- Pandas
- Plotly
- Joblib
- Scikit-learn
- XGBoost

### Setup Instructions
1. **Clone the repository** (or download the source code):
   ```bash
   git clone https://github.com/your-repo/credit-card-dashboard.git
   cd credit-card-dashboard
   ```
2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure
```
credit-card-dashboard/
│-- app.py                   # Main Streamlit application
│-- models/
│   ├── xgboost_model.pkl    # Pre-trained XGBoost model
│-- data/
│   ├── credit_1.csv         # Transaction dataset
│   ├── test_data.csv        # Test dataset for predictions
│-- visualization/
│   ├── daily_spending_trends.png
│   ├── payment_methods.png
│-- requirements.txt         # Dependencies
│-- README.md                # Project documentation
```

## 📖 Usage Guide
1. **Select Customer ID** – Choose a specific customer to analyze spending.
2. **Filter by Category & Date Range** – Narrow down insights based on transaction category.
3. **Explore Visualizations** – View spending patterns through various interactive graphs.
4. **Predict Future Spending** – Use the forecasting tool to project spending trends.
5. **Analyze Fraud Risks** – Check fraud percentage in transaction history.

## 📊 Data Sources & Processing
- **Dataset**: The project uses transaction records with fields like `amt`, `category`, `city_pop`, and `is_fraud`.
- **Preprocessing**: Missing values are handled, dates are converted, and K-Means clustering is applied to assign customer IDs.
- **Machine Learning Model**: XGBoost is trained using historical transaction data to forecast future spending.

## 🛠️ Technical Details
- **Backend**: Python, Streamlit
- **Machine Learning**: XGBoost for forecasting
- **Visualization**: Plotly for interactive charts
- **Clustering**: K-Means to group customers

## 🤝 Contribution
Want to improve this project? Feel free to contribute!
1. Fork the repository.
2. Create a feature branch (`git checkout -b new-feature`).
3. Commit changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin new-feature`).
5. Open a Pull Request.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 📬 Contact
For any queries or suggestions, reach out via **email@example.com** or open an issue on GitHub.

---

🔹 **Developed with ❤️ using Streamlit & Machine Learning** 🚀
