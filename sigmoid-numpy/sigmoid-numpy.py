import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    exp = lambda a: 1/(1 + np.exp((-1) * a))
    f = np.frompyfunc(exp, 1, 1)
    return f(x).astype(float)