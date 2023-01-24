import torch
from market_data import download_price_history
from model import LSTMModel, LSTMModelDefinition
from plots import plot_predict_unseen, plot_train_vs_test, plot_predictions_vs_actual, plot_predictions_vs_actual_zoomed

config = {
    "symbol": "TSLA",
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
    },
    "training": {
        "device": "cpu",  # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 50,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
}

symbol = config["symbol"]

price_history = download_price_history(symbol)
print(f'Loaded {price_history.size()} data points for {symbol}, from {price_history.first_date()} to {price_history.last_date()}')
price_history.plot()

split_ratio = config["data"]["train_split_size"]
window_size = config["data"]["window_size"]

# normalize
lstm_data = price_history.to_lstm_data(split_ratio, window_size)
lstm_data.plot(price_history)

batch_size = config["training"]["batch_size"]
hw_device = config["training"]["device"]

model = LSTMModel(
    LSTMModelDefinition(output_size=1),
    hw_device,
    config["training"]["learning_rate"],
    config["training"]["scheduler_step_size"]
)

model.learn(
    config["training"]["num_epoch"],
    lstm_data.training_dataloader(batch_size),
    lstm_data.testing_dataloader(batch_size)
)

# here we re-initialize dataloader so the data isn't shuffled, so we can plot the values by date
training_dataloader = lstm_data.training_dataloader(batch_size, shuffle=False)
testing_dataloader = lstm_data.testing_dataloader(batch_size, shuffle=False)

predicted_train = model.make_predictions(training_dataloader)
predicted_test = model.make_predictions(testing_dataloader)

plot_predictions_vs_actual(lstm_data, predicted_train, predicted_test, price_history)

plot_predictions_vs_actual_zoomed(lstm_data, predicted_test, price_history.dates)

# predict the closing price of the next trading day

x = torch.tensor(lstm_data.data_x_unseen).float().to(hw_device).unsqueeze(0).unsqueeze(2)  # this is the data type and shape required, [batch, sequence, feature]
prediction = model.make_prediction(x)
prediction = prediction.cpu().detach().numpy()

plot_predict_unseen(lstm_data.scaler, lstm_data.data_y_test, predicted_test, prediction, price_history)
