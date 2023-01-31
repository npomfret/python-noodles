import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from datetime import datetime, timedelta
from typing import List, Any
from nptyping import NDArray, Shape, Float

CONFIG = {
    "xticks_interval": 90,  # show a date every 90 days
    "color_actual": "#001f3f",
    "color_train": "#3D9970",
    "color_test": "#0074D9",
    "color_pred_train": "#3D9970",
    "color_pred_test": "#0074D9",
    "color_pred_unseen": "#FF4136",
}


def plot_raw_prices(price_history):
    dates = price_history.dates

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(dates, price_history.prices, color=(CONFIG["color_actual"]), label=f'EOD Prices for {price_history.symbol}')

    title = f'Daily close price for {price_history.symbol}, "from " + {price_history.first_date()} to {price_history.last_date()}'

    ticks_interval = CONFIG["xticks_interval"]
    num_data_points = price_history.size()
    xticks = [dates[i] if ((i % ticks_interval == 0 and (num_data_points - i) > ticks_interval) or i == num_data_points - 1) else None for i in range(num_data_points)]  # make x ticks nice

    x = np.arange(0, len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.title(title)
    plt.grid(b=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()


def plot_train_vs_test(price_history, lstm_data):
    # prepare data for plotting
    to_plot_data_y_train: NDArray = np.zeros(price_history.size())
    to_plot_data_y_test: NDArray = np.zeros(price_history.size())

    y_train_start = lstm_data.window_size
    y_train_end = lstm_data.split_index + lstm_data.window_size
    to_plot_data_y_train[y_train_start:y_train_end] = lstm_data.unscale(lstm_data.data_y_train)

    y_test_start = y_train_end
    to_plot_data_y_test[y_test_start:] = lstm_data.unscale(lstm_data.data_y_test)

    to_plot_data_y_train = np.where(to_plot_data_y_train == 0, None, to_plot_data_y_train)
    to_plot_data_y_test = np.where(to_plot_data_y_test == 0, None, to_plot_data_y_test)

    ## plots
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(price_history.dates, to_plot_data_y_train, label="Prices (train)", color=CONFIG["color_train"])
    plt.plot(price_history.dates, to_plot_data_y_test, label="Prices (testing)", color=CONFIG["color_test"])

    title = "Daily close prices for " + price_history.symbol + " - showing training and testing data"

    ticks_interval = CONFIG["xticks_interval"]
    xticks = [price_history.dates[i] if ((i % ticks_interval == 0 and (price_history.size() - i) > ticks_interval) or i == price_history.size() - 1) else None for i in range(price_history.size())]  # make x ticks nice

    x = np.arange(0, len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.title(title)
    plt.grid(b=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()


def plot_predictions_vs_actual(price_history, lstm_data, predicted_train: NDArray, predicted_test: NDArray):
    # prepare data for plotting

    to_plot_data_y_train_pred: NDArray = np.zeros(price_history.size())
    to_plot_data_y_val_pred: NDArray = np.zeros(price_history.size())

    to_plot_data_y_train_pred[lstm_data.window_size:lstm_data.split_index + lstm_data.window_size] = lstm_data.unscale(predicted_train)
    to_plot_data_y_val_pred[lstm_data.split_index + lstm_data.window_size:] = lstm_data.unscale(predicted_test)

    to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
    to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(price_history.dates, price_history.prices, label="Actual prices", color=CONFIG["color_actual"])
    plt.plot(price_history.dates, to_plot_data_y_train_pred, label="Predicted prices (train)", color=CONFIG["color_pred_train"])
    plt.plot(price_history.dates, to_plot_data_y_val_pred, label="Predicted prices (validation)", color=CONFIG["color_pred_test"])

    title = "Compare predicted prices to actual prices"

    ticks_interval = CONFIG["xticks_interval"]
    xticks = [price_history.dates[i] if ((i % ticks_interval == 0 and (price_history.size() - i) > ticks_interval) or i == price_history.size() - 1) else None for i in range(price_history.size())]  # make x ticks nice

    x = np.arange(0, len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.title(title)
    plt.grid(b=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()


def plot_predictions_vs_actual_zoomed(price_history, lstm_data, predicted_val: NDArray):
    dates: List[str] = price_history.dates

    ticks_interval = CONFIG["xticks_interval"]

    # prepare data for plotting the zoomed in view of the predicted prices vs. actual prices
    to_plot_data_y_val_subset = lstm_data.unscale(lstm_data.data_y_test)
    to_plot_predicted_val = lstm_data.unscale(predicted_val)
    to_plot_data_date = dates[lstm_data.split_index + lstm_data.window_size:]

    # plots
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(to_plot_data_date, to_plot_data_y_val_subset, label="Actual prices", color=CONFIG["color_actual"])
    plt.plot(to_plot_data_date, to_plot_predicted_val, label="Predicted prices (validation)", color=CONFIG["color_pred_test"])
    xticks = [to_plot_data_date[i] if ((i % int(ticks_interval / 5) == 0 and (len(to_plot_data_date) - i) > ticks_interval / 6) or i == len(to_plot_data_date) - 1) else None for i in range(len(to_plot_data_date))]  # make x ticks nice

    title = "Zoom in to examine predicted price on validation data portion"

    x = np.arange(0, len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.title(title)
    plt.grid(b=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()


def plot_predict_unseen(price_history, lstm_data, predicted_val: NDArray, unseen_prediction: [(Any, 1)]):
    date_object = datetime.strptime(price_history.last_date(), "%Y-%m-%d")
    next_day = date_object + timedelta(days=1)
    next_day_string = next_day.strftime("%Y-%m-%d")

    # prepare plots
    plot_range = 25
    to_plot_data_y_test: NDArray = np.zeros(plot_range)
    to_plot_data_y_test[:plot_range - 1] = lstm_data.unscale(lstm_data.data_y_test)[-plot_range + 1:]
    to_plot_data_y_test = np.where(to_plot_data_y_test == 0, None, to_plot_data_y_test)

    to_plot_data_y_test_pred: NDArray = np.zeros(plot_range)
    to_plot_data_y_test_pred[:plot_range - 1] = lstm_data.unscale(predicted_val)[-plot_range + 1:]
    to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

    to_plot_data_y_unseen_pred: NDArray = np.zeros(plot_range)
    to_plot_data_y_unseen_pred[plot_range - 1] = lstm_data.unscale(unseen_prediction)
    to_plot_data_y_unseen_pred = np.where(to_plot_data_y_unseen_pred == 0, None, to_plot_data_y_unseen_pred)

    # plot
    plot_date_test = price_history.dates[-plot_range + 1:]
    plot_date_test.append(f'>>{next_day_string}<<')
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(plot_date_test, to_plot_data_y_test, label="Actual prices", marker=".", markersize=10, color=CONFIG["color_actual"])
    plt.plot(plot_date_test, to_plot_data_y_test_pred, label="Past predicted prices", marker=".", markersize=10, color=CONFIG["color_pred_test"])
    plt.plot(plot_date_test, to_plot_data_y_unseen_pred, label="Predicted price for next day", marker=".", markersize=20, color=CONFIG["color_pred_unseen"])
    plt.title(f'Predicting the close price of the next trading day ({next_day_string})')
    plt.grid(b=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()

    print(f'Predicted close price of the next trading day ({next_day_string}): {round(to_plot_data_y_unseen_pred[plot_range - 1], 2)}')
