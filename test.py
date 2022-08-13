import torch
import math

import timeit

import util


util.show_backend_info()


def calculate(dtype, device):
    # Create random input and output data
    x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    y = torch.sin(x)

    # Randomly initialize weights
    a = torch.randn((), device=device, dtype=dtype)
    b = torch.randn((), device=device, dtype=dtype)
    c = torch.randn((), device=device, dtype=dtype)
    d = torch.randn((), device=device, dtype=dtype)

    learning_rate = 1e-6
    for t in range(2000):
        # Forward pass: compute predicted y
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        # Compute and print loss
        loss = (y_pred - y).pow(2).sum().item()
        if t % 100 == 99:
            print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred * x).sum()
        grad_c = (grad_y_pred * x ** 2).sum()
        grad_d = (grad_y_pred * x ** 3).sum()

        # Update weights using gradient descent
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d

    print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')


dtype = torch.float

print('Calculating on CPU...')
device = torch.device('cpu')
t0 = timeit.Timer(stmt='calculate(dtype, device)', 
                  setup='from __main__ import calculate',
                  globals={'dtype': dtype, 'device': device})

print('Calculating on GPU...')
device = torch.device(util.get_available_device())
t1 = timeit.Timer(stmt='calculate(dtype, device)', 
                  setup='from __main__ import calculate',
                  globals={'dtype': dtype, 'device': device})

print(f'CPU: {t0.timeit(number=1):.3f}s')
print(f'GPU: {t1.timeit(number=1):.3f}s')