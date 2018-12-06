#!/usr/bin/env python3

import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras_custom_layers import ArgMax

from keras import backend as K
import os

# log2(M) bits to encode
M = 2
# One complex channel needs two real outputs
n_channel = 2

# Define network structure
from_channel = Input(shape=(n_channel,),name='input')
inner_layer = Dense(M, activation='relu')(from_channel)
inner_layer = Dense(M, activation='softmax')(inner_layer)
output_bits = ArgMax()(inner_layer)

decoder = Model(from_channel, output_bits)
# We need an extra identity in tensorflow for the output placeholder
tf.identity(output_bits, name='output')
print(decoder.summary())

# Export directory
export_dir = os.path.dirname(os.path.realpath(__file__))
# Load weights from Keras model
decoder.load_weights(export_dir + '/export/keras_weights_decoder.h5')
# Now save as tensorflow model
sess = K.get_session()
saver = tf.train.Saver()
# Write metagraph model
os.makedirs(export_dir + '/export/decoder', exist_ok=True)
saver.save(sess, export_dir + '/export/decoder/tf_model')
# Write graph for tensorboard
tf.summary.FileWriter(export_dir + '/export/decoder',sess.graph)
