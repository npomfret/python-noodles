import torch
from torch.utils.data import DataLoader
from market_data import download_price_history, create_windowed_data, create_output_array, Normalizer, TimeSeriesDataset
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
scaler = Normalizer()
normalized_prices = scaler.fit_transform(price_history.prices)

data_x, data_x_unseen = create_windowed_data(normalized_prices, window_size)
data_y = create_output_array(normalized_prices, window_size=window_size)

if len(data_x) != len(data_y):
    raise ValueError('x and y are not same length')

# split dataset, early stuff for training, later stuff for testing

split_index = int(data_y.shape[0] * split_ratio)
data_x_train = data_x[:split_index]
data_x_test = data_x[split_index:]
data_y_train = data_y[:split_index]
data_y_test = data_y[split_index:]

plot_train_vs_test(window_size, split_index, scaler, data_y_train, data_y_test, price_history)

dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
dataset_test = TimeSeriesDataset(data_x_test, data_y_test)

print(f'Train data shape, x: {dataset_train.x.shape}, y: {dataset_train.y.shape}')
print(f'Testing data shape, x: {dataset_test.x.shape}, y: {dataset_test.y.shape}')

batch_size = config["training"]["batch_size"]
hw_device = config["training"]["device"]

training_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
testing_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

model = LSTMModel(
    LSTMModelDefinition(output_size=1),
    hw_device,
    config["training"]["learning_rate"],
    config["training"]["scheduler_step_size"]
)

model.learn(
    config["training"]["num_epoch"],
    training_dataloader,
    testing_dataloader
)

# here we re-initialize dataloader so the data isn't shuffled, so we can plot the values by date
training_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
testing_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

predicted_train = model.make_predictions(training_dataloader)
predicted_test = model.make_predictions(testing_dataloader)

plot_predictions_vs_actual(window_size, split_index, scaler, predicted_train, predicted_test, price_history)

plot_predictions_vs_actual_zoomed(scaler, data_y_test, predicted_test, price_history.dates, split_index, window_size)

# predict the closing price of the next trading day

x = torch.tensor(data_x_unseen).float().to(hw_device).unsqueeze(0).unsqueeze(2)  # this is the data type and shape required, [batch, sequence, feature]
prediction = model.make_prediction(x)
prediction = prediction.cpu().detach().numpy()

plot_predict_unseen(scaler, data_y_test, predicted_test, prediction, price_history.dates)
