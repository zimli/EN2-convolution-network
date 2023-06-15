import tensorflow as tf
from tensorflow import keras
from utils import ComplexInit
from tensorflow.keras.layers import Input, Lambda, Concatenate, Add, Multiply,Flatten
from tensorflow.keras import backend as K

class ComplexConv2D(keras.layers.Layer):

    def __init__(self, kw, kh, n_out, sw, sh, activation):
        super(ComplexConv2D, self).__init__()
        self.kw = kw
        self.kh = kh
        self.n_out = n_out
        self.sw = sw
        self.sh = sh
        self.activation = activation

    def build(self, input_shape):
        n_in = input_shape[-1] // 2
        kernel_init = ComplexInit(kernel_size=(self.kh, self.kw),
                                  input_dim=n_in,
                                  weight_dim=2,
                                  nb_filters=self.n_out,
                                  criterion='he')

        self.w = self.add_weight(name='w',
                                 shape=(self.kh, self.kw, n_in, self.n_out*2),
                                 initializer=kernel_init,
                                 trainable=True)

        self.b = self.add_weight(name='b',
                                 shape=(self.n_out*2,),
                                 initializer=keras.initializers.Constant(0.0001),
                                 trainable=True)

    def call(self, inputs):
        kernel_real = self.w[:, :, :, :self.n_out]
        kernel_imag = self.w[:, :, :, self.n_out:]
        cat_kernel_real = tf.concat([kernel_real, -kernel_imag], axis=-2)
        cat_kernel_imag = tf.concat([kernel_imag, kernel_real], axis=-2)
        cat_kernel_complex = tf.concat([cat_kernel_real, cat_kernel_imag], axis=-1)
        conv = tf.nn.conv2d(inputs, cat_kernel_complex, strides=[1, self.sh, self.sw, 1], padding='SAME')
        conv_bias = tf.nn.bias_add(conv, self.b)
        if self.activation:
            act = tf.nn.relu(conv_bias)
            output = act
        else:
            output = conv_bias
        return output

    def get_config(self):
        config = {
            'kw': self.kw,
            'kh': self.kh,
            'n_out': self.n_out,
            'sw': self.sw,
            'sh': self.sh,
            'activation': self.activation
        }
        base_config = super(ComplexConv2D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

def Concatenation(layers):

    return tf.concat(layers,axis=-1)


class FMU(tf.keras.layers.Layer):
    """
    tf.keras.layers.Dense(
        units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
        **kwargs
    )
    """

    def __init__(self, k1,k2,channel1,channel2,channel3):
        super(FMU, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.channel1 = channel1
        self.channel2 = channel2
        self.channel3 = channel3


    def build(self, inputs_shape):
        self.conv1 = ComplexConv2D(kw=self.k1, kh=self.k1, n_out=self.channel1, sw=1, sh=1, activation=True)
        self.conv2 = ComplexConv2D(kw=self.k2, kh=self.k2, n_out=self.channel2, sw=1, sh=1, activation=False)
        self.conv3 = ComplexConv2D(kw=self.k1, kh=self.k1, n_out=self.channel1, sw=1, sh=1, activation=True)
        self.conv4 = ComplexConv2D(kw=self.k2, kh=self.k2, n_out=self.channel3, sw=1, sh=1, activation=True)
        super(FMU, self).build(inputs_shape)

    def call(self, inputs):
        out_in1 = self.conv1(inputs)
        out_in2 = self.conv2(out_in1)
        out_in =Add()([inputs,out_in2])
        out_out1 = self.conv3(inputs)
        out_out = self.conv4(out_out1)
        out = Lambda(Concatenation)([out_in,out_out])

        return out

    def get_config(self):
        config = {
            'k1': self.k1,
            'k2': self.k2,
            'channel1': self.channnel1,
            'channel2': self.channnel2,
            'channel3': self.channnel3,
        }
        base_config = super(Mix, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))