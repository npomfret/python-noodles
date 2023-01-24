import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from market_data import download_price_history
from plots import plot_predict_unseen, plot_train_vs_test, plot_predictions_vs_actual, plot_predictions_vs_actual_zoomed

config = {
    "symbol": "TSLA",
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
    },
    "model": {
        "input_size": 1,  # since we are only using 1 feature, close price
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
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


class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu) / self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x * self.sd) + self.mu


# normalize
scaler = Normalizer()
normalized_prices = scaler.fit_transform(close_prices)


def create_windowed_data(x, window_size):
    number_of_observations = x.shape[0]
    number_of_output_rows = number_of_observations - window_size + 1

    # https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html
    windowed_data = np.lib.stride_tricks.as_strided(
        x,
        shape=(number_of_output_rows, window_size),
        strides=(x.strides[0], x.strides[0]),
        writeable=False
    )

    # 'output' is a 2D view of the 1D input array 'x',
    # each entry is of width 'window_size' and is effectively a shifted copy of the previous entry
    # [
    #     [1,2,3,4,5],
    #     [2,3,4,5,6],
    #     [3,4,5,6,7],
    #     ...

    data = windowed_data[:-1]
    row_with_no_tomorrow = windowed_data[-1]

    return data, row_with_no_tomorrow


def create_output_array(x, window_size):
    # use the next day as label
    output = x[window_size:]
    return output


window_size = config["data"]["window_size"]
data_x, data_x_unseen = create_windowed_data(normalized_prices, window_size)
data_y = create_output_array(normalized_prices, window_size=window_size)

if len(data_x) != len(data_y):
    raise ValueError('x and y are not same length')

# split dataset, early stuff for training, later stuff for testing

split_index = int(data_y.shape[0] * config["data"]["train_split_size"])
data_x_train = data_x[:split_index]
data_x_val = data_x[split_index:]
data_y_train = data_y[:split_index]
data_y_val = data_y[split_index:]

plot_train_vs_test(window_size, split_index, scaler, data_y_train, data_y_val, dates, symbol)


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        number_of_features = 1
        number_of_samples = len(x)
        if number_of_samples != len(y):
            raise ValueError('x and y are not same length')
        number_of_time_steps = len(x[0])

        # in our case, we have only 1 feature, so we need to convert `x` into [batch, sequence, features] for LSTM
        x = np.expand_dims(x, 2)
        if x.shape != (number_of_samples, number_of_time_steps, number_of_features):
            raise ValueError('x is wrong shape for LSTM')

        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

print(f'Train data shape, x: {dataset_train.x.shape}, y: {dataset_train.y.shape}')
print(f'Validation data shape, x: {dataset_val.x.shape}, y: {dataset_val.y.shape}')

batch_size = config["training"]["batch_size"]

train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers * hidden_layer_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]

        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:, -1]


def run_epoch(dataloader, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]

        x = x.to(config["training"]["device"])
        y = y.to(config["training"]["device"])

        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / batchsize)

    lr = scheduler.get_last_lr()[0]

    return epoch_loss, lr


train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

model = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"],
                  num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
model = model.to(config["training"]["device"])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

for epoch in range(config["training"]["num_epoch"]):
    loss_train, lr_train = run_epoch(train_dataloader, is_training=True)
    loss_val, lr_val = run_epoch(val_dataloader)
    scheduler.step()

    print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
          .format(epoch + 1, config["training"]["num_epoch"], loss_train, loss_val, lr_train))

# here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date

train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

model.eval()

# predict on the training data, to see how well the model managed to learn and memorize

predicted_train = np.array([])

for idx, (x, y) in enumerate(train_dataloader):
    x = x.to(config["training"]["device"])
    out = model(x)
    out = out.cpu().detach().numpy()
    predicted_train = np.concatenate((predicted_train, out))

# predict on the validation data, to see how the model does

predicted_val = np.array([])

for idx, (x, y) in enumerate(val_dataloader):
    x = x.to(config["training"]["device"])
    out = model(x)
    out = out.cpu().detach().numpy()
    predicted_val = np.concatenate((predicted_val, out))

plot_predictions_vs_actual(window_size, split_index, scaler, predicted_train, predicted_val, dates, close_prices)

plot_predictions_vs_actual_zoomed(scaler, data_y_val, predicted_val, dates, split_index, window_size)

# predict the closing price of the next trading day

model.eval()

x = torch.tensor(data_x_unseen).float().to(config["training"]["device"]).unsqueeze(0).unsqueeze(2)  # this is the data type and shape required, [batch, sequence, feature]
prediction = model(x)
prediction = prediction.cpu().detach().numpy()

plot_predict_unseen(scaler, data_y_val, predicted_val, prediction, dates)
