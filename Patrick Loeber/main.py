import torch
import numpy as np

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([4, 8, 12, 16], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


def forward(x):
    return w * x


def loss(y, y_predicted):
    return ((y_predicted - y) ** 2).mean()



print(f'prediction before training: f(5) = {forward(5):.3f} (it should be 20)')

# training
learning_rate = 0.01
n_iters = 50

for epoch in range(n_iters):
    y_pred = forward(X)

    l = loss(Y, y_pred)
    l.backward()

    if(epoch % 5 == 0):
        print (f'{epoch}: w = {w:.3f}, loss = {l:.8f}')

    with torch.no_grad():
        w -= learning_rate * w.grad

    w.grad.zero_()

print(f'prediction after training: f(5) = {forward(5):.3f}')
