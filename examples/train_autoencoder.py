#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model
from keras.layers.normalization import BatchNormalization

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from keras.constraints import max_norm
from keras.constraints import unit_norm
from keras.constraints import min_max_norm

from keras import backend as K
import os

# log2(M) bits to encode
M = 2
# Training dataset size
N = 100000
# Validation dataset size
N_val = 100000
# One complex channel needs two real outputs
n_channel = 2
# SNR
EsN0_dB = 5
EsN0 = np.power(10,EsN0_dB/10)

# random labels (bits)
labels = np.random.randint(M,size=N)
labels_val = np.random.randint(M,size=N)
# Generate one hot encoded dataset
data = np.zeros([N,M])
data_val = np.zeros([N,M])
# one hot encoding
for idx, val in enumerate(labels):
    data[idx][val] = 1
for idx, val in enumerate(labels):
    data_val[idx][val] = 1

# Define the encoder structure
input_bits = Input(dtype='float32',shape=(M,))
encoder = Dense(M, activation='relu')(input_bits)
encoder = Dense(n_channel, activation='linear')(encoder)
to_channel = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, 
                                beta_initializer='zeros', gamma_initializer='ones', 
                                moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                beta_regularizer=None, gamma_regularizer=None, 
                                beta_constraint=min_max_norm(min_value=0, max_value=0),
                                gamma_constraint=unit_norm(axis=-1)) (encoder)

# Gaussian channel
channel = GaussianNoise(1.0/(np.sqrt(2*EsN0)))(to_channel)

# Define the decoder structure
decoder = Dense(M, activation='relu')(channel)
output_bits = Dense(M, activation='softmax')(decoder)

# Autoencoder consists of encoder, channel and decoder
autoencoder = Model(input_bits, output_bits)

# Optimizer and loss function
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')
print(autoencoder.summary())

# train the network
history = autoencoder.fit(data, data,
                          epochs=15,
                          batch_size=500,
                          validation_data=(data_val, data_val))

# plot the training history
print(history.history)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

# Extract encoder part from trained autoencoder model
encoder = Model(input_bits, to_channel)

# Extract decoder part from trained autoencoder model
from_channel = Input(shape=(n_channel,))
inner_layer = autoencoder.layers[-2](from_channel)
output_bits = autoencoder.layers[-1](inner_layer)
decoder = Model(from_channel, output_bits)

# Constellation diagram
scatter = encoder.predict(data_val)
plt.scatter(scatter[:,0],scatter[:,1])
circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
ax = plt.gca()
ax.add_artist(circ)
plt.ylabel('I')
plt.xlabel('Q')
plt.axis('equal')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.grid()
plt.show()

# Export encoder and decoder models to file
export_dir = os.path.dirname(os.path.realpath(__file__))
os.makedirs(export_dir + '/export', exist_ok=True)
encoder.save(export_dir + '/export/keras_weights_encoder.h5')
decoder.save(export_dir + '/export/keras_weights_decoder.h5')
