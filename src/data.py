import numpy as np
import pandas as pd

from scipy.special import expit  # sigmoid


def get_toy_dataset(N, D):
    w_true = np.random.randn(1)
    X_data = np.random.randn(N, D)
    p = expit(np.dot(X_data, w_true))
    y_data = np.array([np.random.binomial(1, i) for i in p])
    return X_data, y_data, X_data, y_data


def get_dataset(N=None, D=None, make_balanced=False):
    dat = pd.read_csv("../data/train-1.csv", nrows=N)

    if make_balanced:
        dat1 = dat[dat['target'] == 1].reset_index(drop=True)
        dat0 = dat[dat['target'] == 0].reset_index(drop=True)
        if len(dat0) > len(dat1):
            dat0 = dat0.ix[:len(dat1) - 1]
        else:
            dat1 = dat1.ix[:len(dat0) - 1]
        dat = pd.concat([dat0, dat1]).reset_index(drop=True)

    X_data = dat[[col for col in dat.columns if 'feature' in col]]

    if D:
        X_data = X_data[X_data.columns[:D]].values
    else:
        X_data = X_data.values

    y_data = dat['target'].values

    datt = pd.read_csv("../data/test-1.csv", nrows=N)
    X_test = datt[[col for col in datt.columns if 'feature' in col]]

    if D:
        X_test = X_test[X_test.columns[:D]].values
    else:
        X_test = X_test.values

    y_test = datt['target'].values

    return X_data, y_data, X_test, y_test


def generator(arrays, batch_size):
    """Generate batches, one with respect to each array's first axis."""
    starts = [0] * len(arrays)  # pointers to where we are in iteration
    while True:
        batches = []
        for i, array in enumerate(arrays):
            start = starts[i]
            stop = start + batch_size
            diff = stop - array.shape[0]
            if diff <= 0:
                batch = array[start:stop]
                starts[i] += batch_size
            else:
                batch = np.concatenate((array[start:], array[:diff]))
                starts[i] = diff
            batches.append(batch)
        yield batches
