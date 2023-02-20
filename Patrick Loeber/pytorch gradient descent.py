import torch
import numpy as np
import torch.nn as nn

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[4], [8], [12], [16]], dtype=torch.float32)

n_samples, n_features = X.shape
print(f'n_samples: {n_samples}')
print(f'n_features: {n_features}')

input_size = n_features
output_size = n_features
model = nn.Linear(input_size, output_size)

x_test = torch.tensor([5], dtype=torch.float32)
print(f'prediction before training: f(5) = {model(x_test).item():.3f} (it should be 20)')

# training
learning_rate = 0.01
n_iters = 100
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    y_pred = model(X)

    loss = loss_fn(Y, y_pred)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 5 == 0:
        [w, b] = model.parameters()
        print(f'{epoch}: w = {w[0][0].item():.3f}, loss = {loss:.8f}')

print(f'prediction after training: f(5) = {model(x_test).item():.3f}')
