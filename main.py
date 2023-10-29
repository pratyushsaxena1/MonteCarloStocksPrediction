import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Download the relevant data based on user input
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start = start_date, end = end_date)
    return stock_data

# Based on the times user wants to check, change what data is accounted for 
# For instance, if user wants to predict for next few days, use data from a smaller time frame versus if they want to predict next few years
def calculate_window_size(time_unit):
    if time_unit == 'days':
        return 5
    elif time_unit == 'weeks':
        return 10
    elif time_unit == 'months':
        return 20
    elif time_unit == 'years':
        return 252
    elif time_unit == 'decades':
        return 2520
    else:
        print("Invalid time unit. Please choose 'days', 'weeks', 'months', 'years', or 'decades'.")
        return None

def determine_start_date(end_date, num_units, time_unit):
    if time_unit == 'days':
        return pd.to_datetime(end_date) - timedelta(days = num_units)
    elif time_unit == 'weeks':
        return pd.to_datetime(end_date) - timedelta(weeks = num_units)
    elif time_unit == 'months':
        return pd.to_datetime(end_date) - pd.DateOffset(months = num_units)
    elif time_unit == 'years':
        return pd.to_datetime(end_date) - pd.DateOffset(years = num_units)
    elif time_unit == 'decades':
        return pd.to_datetime(end_date) - pd.DateOffset(years = num_units * 10)
    else:
        print("Invalid time unit. Please choose 'days', 'weeks', 'months', 'years', or 'decades'.")
        return None

def simple_moving_average(data, window_size):
    return data.rolling(window = window_size).mean()

# Use a Monte Carlo simulation to predict prices (info about Monte Carlo: https://aws.amazon.com/what-is/monte-carlo-simulation/#:~:text=The%20Monte%20Carlo%20simulation%20provides,or%20hundreds%20of%20risk%20factors)
# Take the average of the 50 possibilities generated- only using 1 simulation would be highly inaccurate, whereas too many simulations removes potential volatility that is bound to be there
def predict_prices(data, time_unit, num_units):
    last_date = data.index[-1]
    window_size = calculate_window_size(time_unit)
    if window_size is None:
        return None
    end_date = (datetime.now() - timedelta(days = 1)).strftime("%Y-%m-%d")
    start_date = determine_start_date(end_date, num_units, time_unit)
    if start_date is None:
        return None
    if time_unit == 'days':
        next_dates = pd.date_range(start = last_date + timedelta(days = 1), periods = num_units, freq = 'B')
    elif time_unit == 'weeks':
        next_dates = pd.date_range(start = last_date + timedelta(days = 1), periods = num_units * 5, freq = 'B')
    elif time_unit == 'months':
        next_dates = pd.date_range(start = last_date + timedelta(days = 1), periods = num_units * 20, freq = 'B')
    elif time_unit == 'years':
        next_dates = pd.date_range(start = last_date + timedelta(days = 1), periods = num_units * 252, freq = 'B')
    elif time_unit == 'decades':
        next_dates = pd.date_range(start = last_date + timedelta(days = 1), periods = num_units * 2520, freq = 'B')
    else:
        print("Invalid time unit. Please choose 'days', 'weeks', 'months', 'years', or 'decades'.")
        return None
    daily_returns = data.pct_change().dropna()
    volatility = np.std(daily_returns)
    simulations = 50
    predicted_prices = []
    for i in range(simulations):
        prices = [data.iloc[-1]]
        for i in range(len(next_dates) - 1):
            daily_return = np.random.normal(0, volatility)
            price = prices[-1] * (1 + daily_return)
            prices.append(price)
        predicted_prices.append(prices)
    predicted_prices_df = pd.DataFrame(np.array(predicted_prices).T, index = next_dates)
    average_predicted_prices = predicted_prices_df.mean(axis = 1)
    return start_date, end_date, average_predicted_prices

# Get user input from terminal
ticker_symbol = input("Enter the stock symbol (for instance, AAPL): ")
start_date = "2000-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")
stock_data = get_stock_data(ticker_symbol, start_date, end_date)
time_unit = input("Enter the unit of time for prediction (days, weeks, months, years, decades): ")
num_units = int(input("Enter the number of units for prediction: "))
start_date, end_date, average_predicted_prices = predict_prices(stock_data['Close'], time_unit, num_units)

# Plot the results
plt.figure(figsize = (12, 6))
plt.plot(stock_data['Close'][start_date : end_date], label = 'Stock Prices')
plt.plot(average_predicted_prices, label = f'Average Predicted Prices (Next {num_units} {time_unit.capitalize()})', linestyle = '--', color = 'red')
plt.title(f'{ticker_symbol} Stock Prices and Simple Moving Average with Predictions (Monte Carlo)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
