#!/usr/bin/env python3

import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model

from keras.layers.normalization import BatchNormalization

from keras.constraints import max_norm
from keras.constraints import unit_norm
from keras.constraints import min_max_norm

from keras import backend as K
import os

# log2(M) bits to encode
M = 2
# One complex channel needs two real outputs
n_channel = 2

# Define the encoder structure
input_bits = Input(dtype='float32',shape=(M,),name='input')
inner_layer = Dense(M, activation='relu')(input_bits)
inner_layer = Dense(n_channel, activation='linear')(inner_layer)
to_channel = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, 
                                beta_initializer='zeros',   gamma_initializer='ones', 
                                moving_mean_initializer='zeros',  moving_variance_initializer='ones',
                                beta_regularizer=None, gamma_regularizer=None, 
                                beta_constraint=min_max_norm(min_value=0, max_value=0),
                                gamma_constraint=unit_norm(axis=-1)) (inner_layer)

encoder = Model(input_bits, to_channel)
# We need an extra identity in tensorflow for the input placeholder
#output = Lambda(tf.identity,output_shape=(n_channel,), arguments=None, name = 'lambda_output')(to_channel)
tf.identity(to_channel, name='output')

print(encoder.summary())

# Export directory
export_dir = os.path.dirname(os.path.realpath(__file__))
# Load weights from Keras model
encoder.load_weights(export_dir + '/export/keras_weights_encoder.h5')
# Now save as tensorflow model
sess = K.get_session()
saver = tf.train.Saver()
# Write metagraph model
os.makedirs(export_dir + '/export/encoder', exist_ok=True)
saver.save(sess, export_dir + '/export/encoder/tf_model')
# Write graph for tensorboard
tf.summary.FileWriter(export_dir + '/export/encoder',sess.graph)
