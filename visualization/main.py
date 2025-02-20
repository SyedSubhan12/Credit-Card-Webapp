import pandas as pd
import plotly.express as px
import os
import plotly.io as pio

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Convert date column once
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['date'] = df['trans_date_trans_time'].dt.date
    df['month'] = df['trans_date_trans_time'].dt.to_period('M')
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day'] = df['trans_date_trans_time'].dt.day
    
    return df

# Ensure output directory exists
def ensure_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Top spending categories
def top_spending_categories(df, output_dir="."):
    ensure_output_dir(output_dir)

    category_summary = df.groupby('category').agg({'amt': ['sum', 'count']}).reset_index()
    category_summary.columns = ['category', 'total_amount', 'transaction_count']

    fig_amount = px.bar(
        category_summary.sort_values('total_amount', ascending=False).head(10),
        x='total_amount', y='category', orientation='h',
        title='Top Spending Categories by Amount', template='plotly_dark',
        color='total_amount', color_continuous_scale='magma'
    )
    fig_amount.write_html(f"{output_dir}/top_spending_categories_amount.html")
    fig_amount.write_image(f"{output_dir}/top_spending_categories_amount.png")

    fig_volume = px.bar(
        category_summary.sort_values('transaction_count', ascending=False).head(10),
        x='transaction_count', y='category', orientation='h',
        title='Top Spending Categories by Volume', template='plotly_dark',
        color='transaction_count', color_continuous_scale='viridis'
    )
    fig_volume.write_html(f"{output_dir}/top_spending_categories_volume.html")
    fig_volume.write_image(f"{output_dir}/top_spending_categories_volume.png")  # Fixed

# Spending trends over time
def spending_trends_over_time(df, output_dir="."):
    ensure_output_dir(output_dir)

    daily_spending = df.groupby('date')['amt'].sum().reset_index()
    monthly_spending = df.groupby('month')['amt'].sum().reset_index()
    monthly_spending['month'] = pd.to_datetime(monthly_spending['month'].astype(str))

    fig_daily = px.line(
        daily_spending, x='date', y='amt',
        title='Daily Spending Trends', template='plotly_dark', markers=True
    )
    fig_daily.write_html(f"{output_dir}/daily_spending_trends.html")
    fig_daily.write_image(f"{output_dir}/daily_spending_trends.png")

    fig_monthly = px.line(
        monthly_spending, x='month', y='amt',
        title='Monthly Spending Trends', template='plotly_dark', markers=True
    )
    fig_monthly.write_html(f"{output_dir}/monthly_spending_trends.html")
    fig_monthly.write_image(f"{output_dir}/monthly_spending_trends.png")

# Peak spending hours
def peak_spending_hours(df, output_dir="."):
    ensure_output_dir(output_dir)

    hour_spending = df.groupby('hour')['amt'].sum().reset_index()

    fig = px.bar(
        hour_spending, x='hour', y='amt',
        title='Peak Spending Hours', template='plotly_dark',
        color='amt', color_continuous_scale='rdbu'
    )
    fig.write_html(f"{output_dir}/peak_spending_hours.html")
    fig.write_image(f"{output_dir}/peak_spending_hours.png")

# Most common payment methods
def payment_methods(df, output_dir="."):
    ensure_output_dir(output_dir)

    entry_mode_mapping = {0: "CVC", 1: "PIN", 2: "Tap"}
    df['Entry Mode'] = df['Entry Mode'].map(entry_mode_mapping).fillna("Unknown")

    payment_counts = df['Entry Mode'].value_counts().reset_index()
    payment_counts.columns = ['Entry Mode', 'count']

    fig = px.pie(
        payment_counts, values='count', names='Entry Mode',
        hole=0.4, title='Most Common Payment Methods', template='plotly_dark'
    )
    fig.write_html(f"{output_dir}/payment_methods.html")
    fig.write_image(f"{output_dir}/payment_methods.png")

# Spending behavior vs location
def spending_vs_location(df, output_dir="."):
    ensure_output_dir(output_dir)

    fig = px.scatter(
        df, x='city_pop', y='amt', size='amt', color='city_pop',
        title='Spending Behavior vs City Population',
        template='plotly_dark', hover_data=['city_pop', 'amt']
    )
    fig.write_html(f"{output_dir}/spending_vs_location.html")
    fig.write_image(f"{output_dir}/spending_vs_location.png")

# Enhanced animated time series
def enhanced_multi_facet_time_series(df, output_dir="."):
    ensure_output_dir(output_dir)

    df['animation_month'] = df['trans_date_trans_time'].dt.strftime('%Y-%m')

    fig = px.line(
        df, x='trans_date_trans_time', y='amt', color='category',
        animation_frame='animation_month',
        title='Enhanced Animated Time Series by Category',
        template='plotly', line_shape='spline', markers=True
    )
    fig.write_html(f"{output_dir}/enhanced_animated_time_series.html")
    fig.write_image(f"{output_dir}/enhanced_animated_time_series.png")

# Main function
def main(file_path, output_dir="."):
    df = load_data(file_path)
    top_spending_categories(df, output_dir)
    spending_trends_over_time(df, output_dir)
    peak_spending_hours(df, output_dir)
    payment_methods(df, output_dir)
    spending_vs_location(df, output_dir)
    enhanced_multi_facet_time_series(df, output_dir)

# Specify file path and output directory
file_path = 'D:/Credit_Card_Spend_Analysis/data/merged_credit_data.csv'
output_dir = 'D:/Credit_Card_Spend_Analysis/visualization'

main(file_path, output_dir)
