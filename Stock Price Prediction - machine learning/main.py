import math
from pandas_datareader import data as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import tensorflow
import yfinance as yfin

yfin.pdr_override()

plt.style.use('fivethirtyeight')

symbol = 'AAPL'
start_date = '2012-01-01'
end_date = '2019-12-17'
df = web.get_data_yahoo(symbol, start=start_date, end=end_date)

print(df.shape)

plt.figure(figsize=(16, 8))
plt.title('Close Price history')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

data = df.filter(['Close'])
# as an array
dataset = data.values
training_data_len = math.ceil(len(dataset) * 0.8)

# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len, :]
# x is a 2-d array of training data (60 elements wide)
x_train = []
# y is a 1-d array of target data
y_train = []

window_size = 60
train_data_len = len(train_data)
for i in range(window_size, train_data_len):
    # look backwards to creat a window of data up to but not including element 'i'
    window = train_data[i - window_size:i, 0]
    x_train.append(window)

    # target is just the current element (i)
    target = train_data[i, 0]
    y_train.append(target)

# convert the numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

# reshape into the LSTM format...

# num_samples should be same as len(x_train)
num_samples = x_train.shape[0]
# num_time_steps should be same as window_size
num_time_steps = x_train.shape[1]
# num_features -> it's just the close price
num_features = 1
x_train = np.reshape(x_train, (num_samples, num_time_steps, num_features))

# build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], num_features)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=1, epochs=1)

# create the testing dataset from the tail end of the input data
test_data = scaled_data[train_data_len - window_size:, :]
x_test = []
# we do not scale the target data here!
y_test = dataset[train_data_len:, :]
for i in range(window_size, len(test_data)):
    x_test.append(test_data[i - window_size: i, 0])

# again - convert to numpy array
x_test = np.array(x_test)

# again - reshape for LSTM
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], num_features))

# get the models predictions for the x_test dataset
predictions = model.predict(x_test)
# and then scale back so we can compare to unscaled target data
predictions = scaler.inverse_transform(predictions)

# get the root mean squared error (RMSE) - ie the standard deviation of the residuals
rmse = np.sqrt(np.mean((predictions - y_test) ** 2))

# plot it
train = data[:training_data_len].copy()
valid = data[train_data_len:].copy()
valid['Predictions'] = predictions
plt.figure(figsize=(16, 8))
plt.title('model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')
plt.show()

# show the valid and prediction prices
apple_quote = web.get_data_yahoo(symbol, start=start_date, end=end_date)
new_df = apple_quote.filter(['Close'])
last_60_days = new_df[-window_size:].values
last_60_days_scaled = scaler.transform(last_60_days)
X_test = [last_60_days_scaled]
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print(f'price prediction for {end_date}+1: {pred_price}')

end_date_plus_one = '2019-12-10'
apple_quote2 = web.get_data_yahoo(symbol, start=end_date_plus_one, end='2019-12-19')
print(apple_quote2['Close'])