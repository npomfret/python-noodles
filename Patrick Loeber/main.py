import torch
import numpy as np

x = torch.tensor(np.array([3., 4.]), requires_grad=True)
print(f'x: {x}')

y_ = x * 2
y = y_ * 5
print(f'y: {y}')

z = y.mean()
print(f'z (mean): {z}')

z.backward()
print(f'x grad: {x.grad}')