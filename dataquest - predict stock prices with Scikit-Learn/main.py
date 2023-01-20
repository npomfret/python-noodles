import math

import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd
import numpy as np

symbol = "MSFT"
filename = f'data/{symbol}_hist.pkl'

try:
    msft_hist = pd.read_pickle(filename)
    print(f'using cached {symbol} historical data from: {filename}')
except:
    msft = yf.Ticker(symbol)
    msft_hist = msft.history(period="max")
    msft_hist.to_pickle(filename)

data = msft_hist[['Close']]
data = data.rename(columns={'Close': 'T-0_Close'})
# 'price_up_flag' will be a 1 if the close price was greater than the previous close price, 0 otherwise
data['T-0_price_up_flag'] = msft_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])['Close']
data['T-0_daily_return'] = msft_hist.rolling(2).apply(lambda x: x.iloc[1] / x.iloc[0])['Close']
data.dropna(inplace=True)
print(data.head(5))
data['T-0_log_daily_return'] = np.log(data["T-0_daily_return"])
print(data.head(5))

# Target is any row that has seen a 'large' one-day price increase
data['Target'] = (data['T-0_daily_return'] > 1.005).astype(int)
print(data.head(5))

msft_prev = msft_hist.copy()
msft_prev = msft_prev.shift(1)
msft_prev.rename(columns={
    "Close": "T-1_Close",
    "High": "T-1_High",
    "Low": "T-1_Low",
    "Open": "T-1_Open",
    "Volume": "T-1_Volume"
}, inplace=True)
print(msft_prev.head(5))

predictors = ["T-1_Close", "T-1_High", "T-1_Low", "T-1_Open", "T-1_Volume"]

shifted_input_data = msft_prev[predictors]
data = data.join(shifted_input_data).iloc[1:]
# we dropped the first row above because there is no prior day
print(data.tail(5))

# create some synthetic columns
data["weekly_trend"] = data.shift(1).rolling(5).mean()["T-0_price_up_flag"]
data["weekly_ret"] = data.shift(1).rolling(5).sum()["T-0_log_daily_return"]

data.dropna(inplace=True)

full_predictors = predictors + [
    "weekly_trend",
    "weekly_ret"
]


def backtest(data, model, predictor_cols, train_size, test_size=1):
    all_preds = []
    threshold = .6
    row_count = data.shape[0]

    # print(f'row_count: {row_count}, train_size: {train_size}, test_size: {test_size}')

    for i in range(train_size, row_count, test_size):
        train_start = i - train_size
        train_end = i
        train = data.iloc[train_start:train_end].copy()
        if train.shape[0] < train_size:
            break

        training_data = train[predictor_cols]
        # print(f'targets for training:')
        value_counts = train["Target"].value_counts()
        # print(f'{value_counts}')
        model.fit(training_data, train["Target"])

        test_start = i
        test_end = i + test_size
        test = data.iloc[test_start:test_end].copy()
        if test.shape[0] < test_size:
            break

        result = model.predict_proba(test[predictor_cols])
        # print(f'testing...')
        # print(f'result: ${result}')
        # just use the 2nd column
        if result.shape[1] == 2:
            predictions = result[:, 1]
            predictions = pd.Series(predictions, index=test.index)
        else:
            # happens when all inputs targets are either 1 or 0
            if len(value_counts.keys()) != 1:
                raise ValueError('should not get here')

            if 1 in value_counts.keys():
                predictions = pd.Series([1], index=test.index)
            elif 0 in value_counts.keys():
                predictions = pd.Series([0], index=test.index)
            else:
                raise ValueError('should not get here')

        # convert the probabilities to simple buy=1 signals
        predictions[predictions > threshold] = 1
        predictions[predictions <= threshold] = 0
        # print(f'predictions: ${predictions}')
        # print(f'---')

        combined = pd.concat({"Target": test["Target"], "Predictions": predictions}, axis=1)
        all_preds.append(combined)

    return pd.concat(all_preds)


model = RandomForestClassifier(n_estimators=25, min_samples_split=3, random_state=1)
# 60=41%
# 50=43%
# 100=45%
final_predictions = backtest(data.iloc[-(500 * 1):], model, full_predictors, 100, 1)
print('value counts for predictions:')
print(final_predictions["Predictions"].value_counts())

score = precision_score(final_predictions["Target"], final_predictions["Predictions"])
print('score for predictions:')
print(score)
