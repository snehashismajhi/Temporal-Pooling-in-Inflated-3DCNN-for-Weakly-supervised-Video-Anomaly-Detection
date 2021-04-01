from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GRU, LSTM, Input
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.regularizers import l2
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Activation
from keras.layers.core import Dense, Flatten

from keras.layers.convolutional import *




def MIL_model(lam3,n_neuron):
    print("Create Model")
    timesteps=32
    data_dim=1024
    print("Create RGB Model")
    input = Input(shape=(timesteps, data_dim))
    LSTM_1 = LSTM(units=512, activation='tanh', return_sequences=True)(input)
    d1= Dense(n_neuron, init='glorot_normal', W_regularizer=l2(lam3))(LSTM_1)
    d1_dropout= Dropout(0.6)(d1)
    detection_output=Dense(1, init='glorot_normal', W_regularizer=l2(lam3),name='Detection_part', activation='sigmoid')(d1_dropout)
    model = Model(inputs=input, outputs=detection_output)
    return model