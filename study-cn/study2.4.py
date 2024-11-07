import numpy as np
from matplotlib_inline import backend_inline
from d2l import torch as d2l


def f(x):
    return 3 * x ** 2 - 4 * x


for h in 10.0 ** np.arange(-1, -6, -1):
    print(f'h={h:.5f}, numerical limit={(f(1 + h) - f(1)) / h:.5f}')


