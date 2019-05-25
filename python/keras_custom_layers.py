from keras.engine import Layer
from keras.layers import InputSpec
from keras import backend as K
import tensorflow as tf


class UniformNoise(Layer):
    """Apply additive uniform noise

    # Arguments
        minval: Minimum value of the uniform distribution
        maxval: Maximum value of the uniform distribution

    # Input shape
        Arbitrary

    # Output shape
        Same as the input shape.
    """

    def __init__(self, minval=-1.0, maxval=1.0, **kwargs):
        super(UniformNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.minval = minval
        self.maxval = maxval

    def call(self, inputs):
        def noised():
            return inputs + K.random_uniform(shape=K.shape(inputs),
                                             minval=self.minval,
                                             maxval=self.maxval)
        return K.in_train_phase(noised, noised)

    def get_config(self):
        config = {'minval': self.minval, 'maxval': self.maxval}
        base_config = super(UniformNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ArgMax(Layer):
    def __init__(self, **kwargs):
        super(ArgMax, self).__init__(**kwargs)

    def call(self, inputs):
        return K.argmax(inputs)
