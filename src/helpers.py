import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def visualise(X_data, y_data, w, b, n_samples=10):
    w_samples = w.sample(n_samples)[:, 0].eval()
    b_samples = b.sample(n_samples).eval()
    plt.scatter(X_data[:, 0], y_data)
    inputs = np.linspace(0, 1, num=100)
    for ns in range(n_samples):
        output = sigmoid(inputs * w_samples[ns] + b_samples[ns])
        plt.plot(inputs, output)
    return w_samples
