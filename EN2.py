import tensorflow as tf
from tensorflow import keras
from utils import ComplexInit
from tensorflow.keras.layers import Input, Lambda, Concatenate, Add, Multiply,Flatten
from tensorflow.keras import backend as K

class EN2(tf.keras.layers.Layer):
    '''
    tf.keras.layers.Conv1D(
    filters, kernel_size, strides=1, padding='valid', data_format='channels_last',
    dilation_rate=1, groups=1, activation=None, use_bias=True,
    kernel_initializer='glorot_uniform', bias_initializer='zeros',
    kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, bias_constraint=None, **kwargs
    )
    '''

    def __init__(self,
                 filters,
                 kernel_size,activation,padding = 'same'):
        super(EN2, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.real_Conv1D = tf.keras.layers.Conv1D(filters=self.filters,
                                                  kernel_size=self.kernel_size,
                                                  padding=self.padding,
                                                  activation=self.activation)

        self.imag_Conv1D = tf.keras.layers.Conv1D(filters=self.filters,
                                                  kernel_size=self.kernel_size,
                                                  padding=self.padding,
                                                  activation=self.activation)

    def call(self, inputs):
        real_inputs = inputs[..., 0]
        imag_inputs = inputs[..., 1]
        real_outputs = self.real_Conv1D(real_inputs) - self.imag_Conv1D(imag_inputs)
        imag_outputs = self.imag_Conv1D(real_inputs) + self.real_Conv1D(imag_inputs)
        output = tf.stack([real_outputs, imag_outputs], axis=-1)
        #output = tf.keras.layers.LeakyReLU(alpha=0.1) (output)
        return output

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'activation': self.activation
        }
        base_config = super(complex_Conv1D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))