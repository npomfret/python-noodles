import torch
import numpy as np

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([4, 8, 12, 16], dtype=np.float32)

w = 0.0


def forward(x):
    return w * x


def loss(y, y_predicted):
    return ((y_predicted - y) ** 2).mean()


def gradient(x, y, y_predicted):
    return np.dot(2 * x, y_predicted - y).mean()


print(f'prediction before training: f(5) = {forward(5):.3f} (it should be 20)')

# training
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    y_pred = forward(X)

    l = loss(Y, y_pred)

    if(epoch % 1 == 0):
        print (f'{epoch + 1}: w = {w:.3f}, loss = {l:.8f}')

    dw = gradient(X, Y, y_pred)

    w -= learning_rate * dw

print(f'prediction after training: f(5) = {forward(5):.3f}')
