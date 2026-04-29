import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x = np.asarray(x)
    y = 1 / (1 + np.exp(-x))
    return y

# print([sigmoid(x) for x in x])