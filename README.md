# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 

## AIM:
To implement Moving Average Model and Exponential smoothing Using Python.

## ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
 
## PROGRAM:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

data = pd.read_csv('web_traffic.csv',parse_dates=['Timestamp'], dayfirst=True)

data.set_index('Timestamp', inplace=True)
data.sort_index(inplace=True)

data_hourly = data['TrafficCount'].resample('H').sum()

print("Shape of resampled hourly data:", data_hourly.shape)
print("First 10 rows:")
print(data_hourly.head(10))

plt.figure(figsize=(12, 6))
plt.plot(data_hourly, label='Hourly Traffic Count')
plt.title('Original Web Traffic Data')
plt.xlabel('Date')
plt.ylabel('Traffic Count')
plt.legend()
plt.grid()
plt.show()

rolling_mean_5 = data_hourly.rolling(window=5).mean()
rolling_mean_10 = data_hourly.rolling(window=10).mean()

rolling_mean_5.head(10)
rolling_mean_10.head(20)

plt.figure(figsize=(12, 6))
plt.plot(data_hourly, label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)')
plt.plot(rolling_mean_10, label='Moving Average (window=10)')
plt.title('Moving Average of Traffic Data')
plt.xlabel('Time')
plt.ylabel('Traffic Count')
plt.legend()
plt.grid()
plt.show()

scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data_hourly.values.reshape(-1, 1)).flatten(),
                        index=data_hourly.index)

x = int(len(scaled_data) * 0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

model_add = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=24).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))

ax = train_data.plot(figsize=(12, 6))
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["Train", "Prediction", "Test"])
ax.set_title('Exponential Smoothing Forecast vs Actual')
plt.grid()
plt.show()

rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print(f'RMSE: {rmse:.4f}')

final_model = ExponentialSmoothing(scaled_data, trend='add', seasonal='add', seasonal_periods=24).fit()
forecast_steps = 24
final_forecast = final_model.forecast(steps=forecast_steps)

ax = scaled_data.plot(label='Observed', figsize=(12, 6))
final_forecast.plot(ax=ax, label='Forecast', style='--')
plt.title("Forecast for Next 24 Hours")
plt.ylabel("Scaled Traffic Count")
plt.xlabel("Time")
plt.legend()
plt.grid()
plt.show()
```

## OUTPUT:

### Original Data:
![Screenshot (112)](https://github.com/user-attachments/assets/e5e06f81-9b97-47dc-82df-66d616c00569)

![image](https://github.com/user-attachments/assets/4944dad9-36fe-47d3-94d8-27d5c82759f7)

### Moving Average:
![image](https://github.com/user-attachments/assets/e6bc6424-5e8f-4654-8807-a78cee7d44aa)

### Exponential Smoothing:
![image](https://github.com/user-attachments/assets/569f63c8-f7c1-4471-a52e-7003d3fcc617)

### Prediction:
![image](https://github.com/user-attachments/assets/f47923c3-9133-47ee-83e8-82992ee31626)

### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
