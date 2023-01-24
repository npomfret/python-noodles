import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from market_data import download_price_history, create_windowed_data, create_output_array
from model import LSTMModel, TimeSeriesDataset, Normalizer
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

dates, close_prices = download_price_history(symbol)

# normalize
scaler = Normalizer()
normalized_prices = scaler.fit_transform(close_prices)

window_size = config["data"]["window_size"]
data_x, data_x_unseen = create_windowed_data(normalized_prices, window_size)
data_y = create_output_array(normalized_prices, window_size=window_size)

if len(data_x) != len(data_y):
    raise ValueError('x and y are not same length')

# split dataset, early stuff for training, later stuff for testing

split_index = int(data_y.shape[0] * config["data"]["train_split_size"])
data_x_train = data_x[:split_index]
data_x_test = data_x[split_index:]
data_y_train = data_y[:split_index]
data_y_test = data_y[split_index:]

plot_train_vs_test(window_size, split_index, scaler, data_y_train, data_y_test, dates, symbol)

dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
dataset_test = TimeSeriesDataset(data_x_test, data_y_test)

print(f'Train data shape, x: {dataset_train.x.shape}, y: {dataset_train.y.shape}')
print(f'Testing data shape, x: {dataset_test.x.shape}, y: {dataset_test.y.shape}')

batch_size = config["training"]["batch_size"]
hw_device = config["training"]["device"]


def run_epoch(dataloader, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x_tensor, y_tensor) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        x_tensor = x_tensor.to(hw_device)
        y_tensor = y_tensor.to(hw_device)

        out_tensor = model(x_tensor)
        loss = criterion(out_tensor.contiguous(), y_tensor.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        bs = x_tensor.shape[0]
        epoch_loss += (loss.detach().item() / bs)

    learning_rate = scheduler.get_last_lr()[0]

    return epoch_loss, learning_rate


training_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
testing_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

model = LSTMModel(output_size=1)
model = model.to(hw_device)

criterion = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=config["training"]["learning_rate"],
    betas=(0.9, 0.98),
    eps=1e-9
)
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=config["training"]["scheduler_step_size"],
    gamma=0.1
)

number_of_epochs = config["training"]["num_epoch"]

for epoch in range(number_of_epochs):
    loss_train, lr_train = run_epoch(training_dataloader, is_training=True)
    loss_test, lr_test = run_epoch(testing_dataloader)
    scheduler.step()

    print(f'Epoch[{epoch + 1}/{number_of_epochs}], training loss:{loss_train:.6f}, testing loss:{loss_test:.6f}, lr:{lr_train:.6f}]')

# here we re-initialize dataloader so the data isn't shuffled, so we can plot the values by date

training_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
testing_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# Set the module in evaluation mode
model.eval()


def make_predictions(data_loader):
    res = np.array([])

    for idx, (x, y) in enumerate(data_loader):
        x = x.to(hw_device)
        out = model(x)
        out = out.cpu().detach().numpy()
        res = np.concatenate((res, out))

    return res


predicted_train = make_predictions(training_dataloader)
predicted_test = make_predictions(testing_dataloader)

plot_predictions_vs_actual(window_size, split_index, scaler, predicted_train, predicted_test, dates, close_prices)

plot_predictions_vs_actual_zoomed(scaler, data_y_test, predicted_test, dates, split_index, window_size)

# predict the closing price of the next trading day

model.eval()

x = torch.tensor(data_x_unseen).float().to(hw_device).unsqueeze(0).unsqueeze(2)  # this is the data type and shape required, [batch, sequence, feature]
prediction = model(x)
prediction = prediction.cpu().detach().numpy()

plot_predict_unseen(scaler, data_y_test, predicted_test, prediction, dates)
