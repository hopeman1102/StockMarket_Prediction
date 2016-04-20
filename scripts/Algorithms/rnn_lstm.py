#! /usr/bin/python
'''Trains a LSTM on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF+LogReg.

Notes:

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.datasets import imdb

max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

import os
import sys
import pandas as pd

def _load_data(data, n_prev = 100):  
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev])
        docY.append(data.iloc[i+n_prev])
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.1):  
    """
    This just splits data to training and testing parts
    """
    ntrn = int(round(len(df) * (1 - test_size)))
    
    X_train, y_train = _load_data(df.iloc[0:ntrn])
    X_test, y_test = _load_data(df.iloc[ntrn:])

    return (X_train, y_train), (X_test, y_test)

def rnn_lstm(file_dataframe, test_size=0.1, col="high"):
    print('Loading data...')
    (X_train, y_train), (X_test, y_test) = train_test_split(\
        file_dataframe[col], test_size=0.2)

    print(X_train.shape, y_train.shape)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen, dropout=0.2))
    model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2)) 
    model.add(Dense(1))
    model.add(Activation('sigmoid'))


    model.compile(loss="mean_squared_error", optimizer="rmsprop") 

    print('Train...')
    print(X_train.shape)
    print(y_train.shape)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15)
    score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

    return (score, accuracy)


def main(dir_path, output_dir):
    '''
        Run Pipeline of processes on file one by one.
    '''
    files = os.listdir(dir_path)

    for file_name in files:

        file_dataframe = pd.read_csv(os.path.join(dir_path, file_name))

        print(rnn_lstm(file_dataframe, 0.1, 'high'))

        break

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])


