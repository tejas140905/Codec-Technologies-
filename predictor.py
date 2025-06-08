import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load stock data
ticker = 'AAPL'
df = yf.download(ticker, start='2015-01-01', end='2024-12-31')
df = df[['Close']].dropna()

# Create features
df['Prediction'] = df[['Close']].shift(-30)
X = df[['Close']][:-30]
y = df['Prediction'][:-30]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions
predictions = lr.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Plot
plt.figure(figsize=(12,6))
plt.plot(y_test.values, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.title(f"{ticker} Stock Price Prediction")
pl
