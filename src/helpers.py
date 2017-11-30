import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import edward as ed

import six
import tensorflow as tf


from edward.models import RandomVariable
from edward.util import check_data, get_session

try:
    from edward.models import Bernoulli, Binomial, Categorical, \
        Multinomial, OneHotCategorical
except Exception as e:
    raise ImportError(
        "{0}. Your TensorFlow version is not supported.".format(e))

from sklearn.metrics import log_loss
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

    # plt.title(f'{title_prefix} Parameter Samples vs. Observed Data')

    # plot lines defined by parameter samples
    inputs = np.linspace(X.min(), X.max(), num=500)
    for i in range(n_samples):
        output = inputs * β_samples[i][0] \
            + inputs * β_samples[i][1] + α_samples[i][0]
        ax.plot(inputs, inputs, output)


def get_model_probs(data, metrics='probs', n_samples=500):
    output_key = None
    # seed = None
    sess = get_session()
    if isinstance(metrics, str):
        metrics = [metrics]
    elif not isinstance(metrics, list):
        raise TypeError("metrics must have type str or list.")

    check_data(data)
    if not isinstance(n_samples, int):
        raise TypeError("n_samples must have type int.")

    if output_key is None:
        # Default output_key to the only data key that isn't a placeholder.
        keys = [key for key in six.iterkeys(data) if not
                isinstance(key, tf.Tensor) or "Placeholder" not in key.op.type]
        if len(keys) == 1:
            output_key = keys[0]
        else:
            raise KeyError("User must specify output_key.")
    elif not isinstance(output_key, RandomVariable):
        raise TypeError("output_key must have type RandomVariable.")

    # Create feed_dict for data placeholders that the model conditions
    # on; it is necessary for all session runs.
    feed_dict = {key: value for key, value in six.iteritems(data)
                 if
                 isinstance(key, tf.Tensor) and "Placeholder" in key.op.type}

    # Form true data.
    y_true = data[output_key]
    # Make predictions (if there are any supervised metrics).
    if metrics != ['log_lik'] and metrics != ['log_likelihood']:
        binary_discrete = (Bernoulli, Binomial)
        categorical_discrete = (Categorical, Multinomial, OneHotCategorical)
        total_count = sess.run(
            getattr(output_key, 'total_count', tf.constant(1.)))
        if isinstance(output_key, binary_discrete + categorical_discrete):
            # Average over realizations of their probabilities, then predict
            # via argmax over probabilities.
            probs = [sess.run(output_key.probs, feed_dict) for _ in
                     range(n_samples)]
            probs = np.sum(probs, axis=0) / n_samples
            if isinstance(output_key, binary_discrete):
                # make random prediction whenever probs is exactly 0.5
                random = tf.random_uniform(shape=tf.shape(probs))
                y_pred = tf.round(
                    tf.where(tf.equal(0.5, probs), random, probs))
            else:
                if total_count > 1:
                    raise Exception('todo multinomial')
                    # if len(output_key.sample_shape):
                    #     y_pred = tf.reshape(
                    #         tf.tile(mode, output_key.sample_shape),
                    #         [-1, len(probs)])
                    # else:
                    #     y_pred = mode
                else:
                    y_pred = tf.argmax(probs, len(probs.shape) - 1)
            probs = tf.constant(probs)
        else:
            # Monte Carlo estimate the mean of the posterior predictive.
            y_pred = [sess.run(output_key, feed_dict) for _ in
                      range(n_samples)]
            y_pred = tf.cast(tf.add_n(y_pred), y_pred[0].dtype) / \
                tf.cast(n_samples, y_pred[0].dtype)
        if len(y_true.shape) == 0:
            y_true = tf.expand_dims(y_true, 0)
            y_pred = tf.expand_dims(y_pred, 0)
    return probs


def binary_crossentropy(y_true, y_pred):
    """Binary cross-entropy.
    Args:
      y_ue: tf.Tensor.
        Tensor of 0s and 1s.
      y_pred: tf.Tensor.
        Tensor of real values (logit probabilities), with same shape as
        `y_true`.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true))


def get_y_preds(X_data, y_data, qvars, n_samples=10):
    return [[qvar.eval() for _ in range(n_samples)] for qvar in qvars]


def eval_model(Xx, Yy, nom, probabilities=None,
               dataset=None, qw=None, qb=None):
    if probabilities is None:
        ans = get_y_preds(Xx, Yy, qvars=[qw, qb], n_samples=10000)
        zw, zb = [np.mean(an, axis=0) for an in ans]
        probabilities = sigmoid(Xx.dot(zw) + zb)

    ins = pd.DataFrame(probabilities, columns=['prob'])
    ins['target'] = Yy
    ins['guess'] = ins['prob'].round().clip(0, 1)
    print(nom, '\n------')
    print('acc:', (ins['target'] == ins['guess']).mean())
    print('sk.logloss:', log_loss(ins['target'], ins['prob']))
    sess = get_session()
    print('ed.logloss (corrected):', sess.run(
        binary_crossentropy(ins['target'], -np.log(1 / ins['prob'] - 1))))
    print('ed.logloss (current):',
          sess.run(binary_crossentropy(ins['target'], ins['prob'])))

    if dataset is not None:
        if nom == 'ins':
            eras = dataset.training_data['era']
        else:
            eras = dataset.prediction_data['era']

        big_ins = pd.concat([pd.DataFrame(Xx), eras, ins], axis=1).dropna()
        consist = 0
        for x in big_ins.era.unique():
            lil_ins = big_ins[big_ins.era == x]
            lloss = log_loss(lil_ins['target'], lil_ins['prob'])
            print(x.replace("era", "regime"),
                  '(%s) :' % str(float(len(lil_ins)) / len(big_ins))[:5],
                  (lil_ins['target'] == lil_ins['guess']).mean(),
                  lloss)
            if lloss < -np.log(.5):
                consist += 1

        print('consistency:', float(consist) / len(big_ins.era.unique()))
        print('')
    return ins
