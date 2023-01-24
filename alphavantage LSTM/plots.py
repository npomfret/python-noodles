import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

CONFIG = {
    "xticks_interval": 90,  # show a date every 90 days
    "color_actual": "#001f3f",
    "color_train": "#3D9970",
    "color_val": "#0074D9",
    "color_pred_train": "#3D9970",
    "color_pred_val": "#0074D9",
    "color_pred_test": "#FF4136",
}


def plot_raw_prices(data_date, data_close_price, symbol):
    ticks_interval = CONFIG["xticks_interval"]
    color = CONFIG["color_actual"]

    num_data_points = len(data_date)
    display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points - 1]

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, data_close_price, color=color)
    xticks = [data_date[i] if ((i % ticks_interval == 0 and (num_data_points - i) > ticks_interval) or i == num_data_points - 1) else None for i in range(num_data_points)]  # make x ticks nice
    x = np.arange(0, len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.title("Daily close price for " + symbol + ", " + display_date_range)
    plt.grid(b=None, which='major', axis='y', linestyle='--')
    plt.show()


def plot_train_vs_test(window_size, split_index, scaler, data_y_train, data_y_val, data_date, symbol):
    num_data_points = len(data_date)
    ticks_interval = CONFIG["xticks_interval"]

    # prepare data for plotting
    to_plot_data_y_train = np.zeros(num_data_points)
    to_plot_data_y_val = np.zeros(num_data_points)
    to_plot_data_y_train[window_size:split_index + window_size] = scaler.inverse_transform(data_y_train)
    to_plot_data_y_val[split_index + window_size:] = scaler.inverse_transform(data_y_val)
    to_plot_data_y_train = np.where(to_plot_data_y_train == 0, None, to_plot_data_y_train)
    to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
    ## plots
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, to_plot_data_y_train, label="Prices (train)", color=CONFIG["color_train"])
    plt.plot(data_date, to_plot_data_y_val, label="Prices (validation)", color=CONFIG["color_val"])
    xticks = [data_date[i] if ((i % ticks_interval == 0 and (num_data_points - i) > ticks_interval) or i == num_data_points - 1) else None for i in range(num_data_points)]  # make x ticks nice
    x = np.arange(0, len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.title("Daily close prices for " + symbol + " - showing training and validation data")
    plt.grid(b=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()


def plot_predictions_vs_actual(window_size, split_index, scaler, predicted_train, predicted_val, data_date, data_close_price):
    num_data_points = len(data_date)

    # prepare data for plotting
    ticks_interval = CONFIG["xticks_interval"]

    to_plot_data_y_train_pred = np.zeros(num_data_points)
    to_plot_data_y_val_pred = np.zeros(num_data_points)

    to_plot_data_y_train_pred[window_size:split_index + window_size] = scaler.inverse_transform(predicted_train)
    to_plot_data_y_val_pred[split_index + window_size:] = scaler.inverse_transform(predicted_val)

    to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
    to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, data_close_price, label="Actual prices", color=CONFIG["color_actual"])
    plt.plot(data_date, to_plot_data_y_train_pred, label="Predicted prices (train)", color=CONFIG["color_pred_train"])
    plt.plot(data_date, to_plot_data_y_val_pred, label="Predicted prices (validation)", color=CONFIG["color_pred_val"])
    plt.title("Compare predicted prices to actual prices")
    xticks = [data_date[i] if ((i % ticks_interval == 0 and (num_data_points - i) > ticks_interval) or i == num_data_points - 1) else None for i in range(num_data_points)]  # make x ticks nice
    x = np.arange(0, len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.grid(b=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()


def plot_predictions_vs_actual_zoomed(scaler, data_y_val, predicted_val, data_date, split_index, window_size):
    ticks_interval = CONFIG["xticks_interval"]

    # prepare data for plotting the zoomed in view of the predicted prices vs. actual prices
    to_plot_data_y_val_subset = scaler.inverse_transform(data_y_val)
    to_plot_predicted_val = scaler.inverse_transform(predicted_val)
    to_plot_data_date = data_date[split_index + window_size:]
    # plots
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(to_plot_data_date, to_plot_data_y_val_subset, label="Actual prices", color=CONFIG["color_actual"])
    plt.plot(to_plot_data_date, to_plot_predicted_val, label="Predicted prices (validation)", color=CONFIG["color_pred_val"])
    plt.title("Zoom in to examine predicted price on validation data portion")
    xticks = [to_plot_data_date[i] if ((i % int(ticks_interval / 5) == 0 and (len(to_plot_data_date) - i) > ticks_interval / 6) or i == len(to_plot_data_date) - 1) else None for i in range(len(to_plot_data_date))]  # make x ticks nice
    xs = np.arange(0, len(xticks))
    plt.xticks(xs, xticks, rotation='vertical')
    plt.grid(b=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()


def plot_predict_unseen(scaler, data_y_val, predicted_val, prediction, data_date):
    # prepare plots
    plot_range = 10
    to_plot_data_y_val = np.zeros(plot_range)
    to_plot_data_y_val_pred = np.zeros(plot_range)
    to_plot_data_y_test_pred = np.zeros(plot_range)
    to_plot_data_y_val[:plot_range - 1] = scaler.inverse_transform(data_y_val)[-plot_range + 1:]
    to_plot_data_y_val_pred[:plot_range - 1] = scaler.inverse_transform(predicted_val)[-plot_range + 1:]
    to_plot_data_y_test_pred[plot_range - 1] = scaler.inverse_transform(prediction)
    to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
    to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
    to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)
    # plot
    plot_date_test = data_date[-plot_range + 1:]
    plot_date_test.append("tomorrow")
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(plot_date_test, to_plot_data_y_val, label="Actual prices", marker=".", markersize=10, color=CONFIG["color_actual"])
    plt.plot(plot_date_test, to_plot_data_y_val_pred, label="Past predicted prices", marker=".", markersize=10, color=CONFIG["color_pred_val"])
    plt.plot(plot_date_test, to_plot_data_y_test_pred, label="Predicted price for next day", marker=".", markersize=20, color=CONFIG["color_pred_test"])
    plt.title("Predicting the close price of the next trading day")
    plt.grid(b=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()

    print("Predicted close price of the next trading day:", round(to_plot_data_y_test_pred[plot_range - 1], 2))
