import numpy as np
import matplotlib.pyplot as plt
import edward as ed
# from mpl_toolkits.mplot3d import Axes3D


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def visualise(X_data, y_data, w, b, n_samples=10):
    w_samples = w.sample(n_samples)[:, 0].eval()
    b_samples = b.sample(n_samples).eval()
    plt.scatter(X_data[:, 0], y_data)
    inputs = np.linspace(X_data.min(), X_data.max(), num=100)
    for ns in range(n_samples):
        output = sigmoid(inputs * w_samples[ns] + b_samples[ns])
        plt.plot(inputs, output)
    return w_samples


def compute_mean_absolute_error(y_posterior, X_val_feed_dict, y_val):
    data = {y_posterior: y_val}
    data.update(X_val_feed_dict)
    mae = ed.evaluate('mean_absolute_error', data=data)
    return mae
    # print(f'Mean absolute error on validation data: {mae:1.5}')


def plot_residuals(y_posterior, X_val_feed_dict, title, y_val):
    y_posterior_preds = y_posterior.eval(feed_dict=X_val_feed_dict)
    plt.figure(figsize=(9, 6))
    plt.hist(y_posterior_preds - y_val,
             edgecolor='white', linewidth=1, bins=30, alpha=.7)
    plt.axvline(0, color='#A60628', linestyle='--')
    plt.xlabel('`y_posterior_preds - y_val`', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title(title, fontsize=16)


def visualize_data_fit(X, y, β, α, title_prefix, n_samples=10):
    '''Plot lines generated via samples from parameter dists of the first
    two fixed effects, vs. observed data points.
    Args:
    X (np.array) : A design matrix of observed fixed effects.
    y (np.array) : A vector of observed responses.
    β (ed.RandomVariable) : A multivariate dist of fixed-effect parameters.
    α (ed.RandomVariable) : A univariate dist of the model's intercept term.
    title_prefix (str) : A string to append to the beginning of the title.
    n_samples (int) : The num of lines to plot as drawn from the param dists.
    '''

    # draw samples from parameter distributions
    β_samples = β.sample(n_samples).eval()
    α_samples = α.sample(n_samples).eval()

    # plot the first two dimensions of `X`, vs. `y`
    # fig = plt.figure(figsize=(12, 9))
    ax = plt.axes(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, alpha=.1)

    plt.title(f'{title_prefix} Parameter Samples vs. Observed Data')

    # plot lines defined by parameter samples
    inputs = np.linspace(X.min(), X.max(), num=500)
    for i in range(n_samples):
        output = inputs * β_samples[i][0] \
            + inputs * β_samples[i][1] + α_samples[i][0]
        ax.plot(inputs, inputs, output)
