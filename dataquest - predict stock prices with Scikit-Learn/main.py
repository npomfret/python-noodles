import env
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd
import numpy as np
from fracdiff.sklearn import FracdiffStat

symbol = "MSFT"
filename = f'data/{symbol}_hist.pkl'

try:
    msft_hist = pd.read_pickle(filename)
    print(f'using cached {symbol} historical data from: {filename}')
except:
    msft = yf.Ticker(symbol)
    msft_hist = msft.history(period="max")
    msft_hist.to_pickle(filename)

print(msft_hist.head(5))

data = msft_hist[['Close']]
data = data.rename(columns={'Close': 'prev_Close'})
data['prev_daily_return_partial'] = FracdiffStat().fit_transform(data)
data['prev_pct_change'] = data['prev_Close'].pct_change()
data['prev_log_change'] = np.log(data['prev_pct_change'] + 1)

data = data.shift(1)
data = data.join(msft_hist[['Close']])
data = data.rename(columns={'Close': 'actual_Close'})
print(data.head(5))

data.dropna(inplace=True)
data['actual_daily_return'] = data['actual_Close'] / data['prev_Close']
data['Target'] = (data['actual_daily_return'] > 1.005).astype(int)
print(data.head(10))

full_predictors = [
    "prev_Close",
    "prev_daily_return_partial",
    # "prev_log_change",
    # "prev_pct_change",
]

print(full_predictors)


def backtest(data, model, predictor_cols, train_size, test_size=1):
    all_preds = []
    threshold = .55
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
            # this is the 'normal' execution path
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
final_predictions = backtest(data.iloc[-(1000 * 1):], model, full_predictors, 200, 5)
print('value counts for predictions:')
print(final_predictions["Predictions"].value_counts())

score = precision_score(final_predictions["Target"], final_predictions["Predictions"])
print('score for predictions:')
print(score)
