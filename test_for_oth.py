# example.py

import torch
from torch.nn import Conv2d

torch.manual_seed(0)

shape = (30, 3, 256, 256)
x = torch.linspace(0, 1, shape[0] * shape[1] * shape[2] * shape[3]).view(shape)
layer = Conv2d(3, 1024, kernel_size=(16, 16), stride=(16, 16))

print('cpu:')
print(layer(x[:1]).mean((1, 2, 3)).item())
print(layer(x[:1]).mean((1, 2, 3)).item())
print(layer(x[:2]).mean((1, 2, 3))[:1].item())
print(layer(x[:2]).mean((1, 2, 3))[:1].item())

x = x.to('cuda')
layer = layer.to('cuda')

print('cuda:')
print(layer(x[:1]).mean((1, 2, 3)).item())
print(layer(x[:1]).mean((1, 2, 3)).item())
print(layer(x[:2]).mean((1, 2, 3))[:1].item())
print(layer(x[:2]).mean((1, 2, 3))[:1].item())

print('cuda again:')
print(layer(x[:1]).mean((1, 2, 3)).item())
print(layer(x[:1]).mean((1, 2, 3)).item())
print(layer(x[:2]).mean((1, 2, 3))[:1].item())
print(layer(x[:2]).mean((1, 2, 3))[:1].item())

layer = layer.double()
x = x.double()

print('cuda double precision:')
print(layer(x[:1]).mean((1, 2, 3)).item())
print(layer(x[:1]).mean((1, 2, 3)).item())
print(layer(x[:2]).mean((1, 2, 3))[:1].item())
print(layer(x[:2]).mean((1, 2, 3))[:1].item())
