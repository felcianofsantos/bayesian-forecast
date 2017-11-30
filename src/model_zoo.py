from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from tensorflow import round as tf_round
import keras.backend as K


def mean_pred(y_true, y_pred):
    return K.mean(tf_round(y_pred))


def conv_lstm(features=5):
    model = Sequential()
    model.add(GaussianNoise(.0001, input_shape=(features, 1)))
    model.add(Conv1D(filters=64, kernel_size=5,
                     padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', mean_pred])
    return model
