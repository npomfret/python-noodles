import math

import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd

msft = yf.Ticker("MSFT")
msft_hist = msft.history(period="max")

data = msft_hist[['Close']]
data = data.rename(columns={'Close': 'Actual_Close'})
# 'target' will be a 1 if the close price was greater than the previous close price, 0 otherwise
data['Target'] = msft_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])['Close']

msft_prev = msft_hist.copy()
msft_prev = msft_prev.shift(1)

predictors = ['Close', "High", "Low", "Open", "Volume"]

data = data.join(msft_prev[predictors]).iloc[1:]
# we dropped the first row above because there is no prior day
print(data.head(5))

# model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)
#
# train = data.iloc[:-100]
# test = data.iloc[-100:]
#
# print("learning...")
# model.fit(train[predictors], train["Target"])
#
# preds = model.predict(test[predictors])
# preds = pd.Series(preds, index=test.index)
#
# score = precision_score(test["Target"], preds)
# print(f'score: {score}')
#
# combined = pd.concat({"Target": test["Target"], "Predictions": preds}, axis=1)

# create some synthetic columns
short_mean = data.rolling(10).mean()
medium_mean = data.rolling(30).mean()
long_mean = data.rolling(90).mean()
weekly_trend = data.shift(1).rolling(5).mean()["Target"]

data["short_mean"] = short_mean["Close"] / data["Close"]
data["medium_mean"] = medium_mean["Close"] / data["Close"]
data["long_mean"] = long_mean["Close"] / data["Close"]

data["long_mean_short_mean_ratio"] = data["long_mean"] / data["short_mean"]
data["long_mean_medium_mean_ratio"] = data["long_mean"] / data["medium_mean"]
data["weekly_trend"] = weekly_trend

data["open_close_ratio"] = data["Open"] / data["Close"]
data["high_close_ratio"] = data["High"] / data["Close"]
data["low_close_ratio"] = data["Low"] / data["Close"]

full_predictors = predictors + ["short_mean", "medium_mean", "long_mean", "long_mean_short_mean_ratio",
                                "long_mean_medium_mean_ratio", "open_close_ratio", "high_close_ratio", "low_close_ratio",
                                "weekly_trend"]

model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)


def backtest(data, model, predictors, train_size, test_size=1):
    all_preds = []
    threshold = .6
    row_count = data.shape[0]

    print(f'row_count: {row_count}, train_size: {train_size}, test_size: {test_size}')

    for i in range(train_size, row_count, test_size):
        train_start = i - train_size
        train_end = i
        train = data.iloc[train_start:train_end].copy()
        if train.shape[0] < train_size:
            break

        model.fit(train[predictors], train["Target"])

        test_start = i
        test_end = i + test_size
        test = data.iloc[test_start:test_end].copy()
        if test.shape[0] < test_size:
            break

        result = model.predict_proba(test[predictors])
        # just use the 2nd column
        if result.shape[1] == 2:
            predictions = result[:, 1]
            predictions = pd.Series(predictions, index=test.index)
        else:
            predictions = pd.Series([0], index=test.index)
        # convert the probabilities to simple buy=1 signals
        predictions[predictions > threshold] = 1
        predictions[predictions <= threshold] = 0
        print(f'{predictions}')

        combined = pd.concat({"Target": test["Target"], "Predictions": predictions}, axis=1)
        all_preds.append(combined)

    return pd.concat(all_preds)


final_predictions = backtest(data.iloc[-(500 * 1):], model, full_predictors, 3, 1)
print('value counts for predictions:')
print(final_predictions["Predictions"].value_counts())

score = precision_score(final_predictions["Target"], final_predictions["Predictions"])
print('score for predictions:')
print(score)
