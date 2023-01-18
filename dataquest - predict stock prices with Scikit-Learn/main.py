import math

import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd

msft = yf.Ticker("MSFT")
msft_hist = msft.history(period="max")

data = msft_hist[['Close']]
data = data.rename(columns={'Close': 'Actual_Close'})
data['Target'] = msft_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])['Close']

msft_prev = msft_hist.copy()
msft_prev = msft_prev.shift(1)

predictors = ['Close', "High", "Low", "Open", "Volume"]
full_predictors = predictors + ["weekly_mean", "quarterly_mean", "annual_mean", "annual_weekly_mean",
                                "annual_quarterly_mean", "open_close_ratio", "high_close_ratio", "low_close_ratio",
                                "weekly_trend"]

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
weekly_mean = data.rolling(7).mean()
quarterly_mean = data.rolling(90).mean()
annual_mean = data.rolling(365).mean()
weekly_trend = data.shift(1).rolling(7).mean()["Target"]
data["weekly_mean"] = weekly_mean["Close"] / data["Close"]
data["quarterly_mean"] = quarterly_mean["Close"] / data["Close"]
data["annual_mean"] = annual_mean["Close"] / data["Close"]

data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]
data["weekly_trend"] = weekly_trend

data["open_close_ratio"] = data["Open"] / data["Close"]
data["high_close_ratio"] = data["High"] / data["Close"]
data["low_close_ratio"] = data["Low"] / data["Close"]

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

        predictions = model.predict_proba(test[predictors])[:, 1]
        predictions[predictions > threshold] = 1
        predictions[predictions <= threshold] = 0
        predictions = pd.Series(predictions, index=test.index)
        print(f'{predictions}')

        combined = pd.concat({"Target": test["Target"], "Predictions": predictions}, axis=1)
        all_preds.append(combined)

    all_preds = pd.concat(all_preds)
    print(all_preds["Predictions"].value_counts())
    return all_preds


final_predictions = backtest(data.iloc[-(365 * 3):], model, full_predictors, 300, 5)
score = precision_score(final_predictions["Target"], final_predictions["Predictions"])
print(score)
