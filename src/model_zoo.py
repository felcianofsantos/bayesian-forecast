from keras import regularizers
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, LSTM, Dropout
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling1D
from keras.layers.core import Reshape
from keras.layers.noise import GaussianNoise
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.models import model_from_yaml
from keras.models import load_model
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(tf.round(y_pred))

def conv_lstm(features=5):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5,
                     padding='same', activation='relu',
                     input_shape=(features, 1), ))
    model.add(BatchNormalization())
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', mean_pred])
    return model