import tensorflow as tf
from tensorflow import keras
from utils import ComplexInit
from tensorflow.keras.layers import Input, Lambda, Concatenate, Add, Multiply,Flatten
from tensorflow.keras.applications import VGG16
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

class complex_Conv1D(tf.keras.layers.Layer):
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
        super(complex_Conv1D, self).__init__()
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




def Fourier2(x):

    x_complex = tf.complex(x[..., 0], x[..., 1])
    y_complex = tf.signal.fft2d(x_complex)
    return y_complex

def FFt(x):
    x_complex = tf.complex(x[..., 0], x[..., 1])
    y_complex = tf.signal.fft2d(x_complex)
    y_complex = tf.transpose(y_complex, [0, 2, 1])  # frequency
    return y_complex

def r2c(r):
    c = tf.complex(r[..., 0], r[..., 1])
    return c

def c2r1(c):
    real = tf.math.real(c)
    imag = tf.math.imag(c)
    r = tf.stack([real,imag],axis=-1)
    return r

def FFT_real(x):
    x1 = tf.transpose(x,[0,3,1,2])
    k = tf.signal.fftshift(tf.signal.fft2d(tf.cast((x1),tf.complex64)),[-1,-2])
    #k = tf.signal.ifft2d(tf.signal.ifftshift(tf.cast((x1),tf.complex64),[-1,-2]))
    k1 = tf.transpose(k,[0,2,3,1])
    return k1


def iFourier2(x):
    corrected_complex = tf.signal.ifft2d(x)
    corrected_real = tf.math.real(corrected_complex)
    corrected_imag = tf.math.imag(corrected_complex)
    y_complex = tf.stack([corrected_real, corrected_imag], axis=-1)
    return y_complex

def add_dc_layer1(x,features,mask):
    first_layer = features
    feature_kspace = Lambda(Fourier2)(first_layer)
    projected_kspace = Multiply()([feature_kspace, mask])

    last_layer = x
    gene_kspace = Lambda(r2c)(last_layer)
    gene_kspace = Multiply()([gene_kspace, (1.0 - mask)])

    corrected_kspace = Add()([projected_kspace, gene_kspace])


    # inverse fft

    return corrected_kspace


def add_dc_layer2(x, features, mask):
    # add dc connection for each block
    first_layer = features
    feature_kspace = Lambda(Fourier2)(first_layer)
    projected_kspace = Multiply()([feature_kspace, mask])

    # get output and input
    last_layer = x
    gene_kspace = Lambda(Fourier2)(last_layer)
    gene_kspace = Multiply()([gene_kspace, (1.0 - mask)])

    corrected_kspace = Add()([projected_kspace, gene_kspace])

    # inverse fft
    corrected_real_concat = Lambda(iFourier2)(corrected_kspace)

    return corrected_real_concat

def Concatenation(layers):

    return tf.concat(layers,axis=-1)




def Concatenation_1d(layers):
    return tf.concat(layers, axis=2)


class Mix(tf.keras.layers.Layer):
    """
    tf.keras.layers.Dense(
        units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
        **kwargs
    )
    """

    def __init__(self, k1,k2,channel1,channel2,channel3):
        super(Mix, self).__init__()
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
        super(Mix, self).build(inputs_shape)

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





def getModel(img_width, img_height, channels):
    inputs = Input(shape=[img_width, img_height, channels], name='zero-filling')  # 96*96*1
    inputs_k_c = Lambda(Fourier2)(inputs)
    inputs_k_r = Lambda(c2r1)(inputs_k_c)
    mask = Input(shape=[img_width, img_height], name='undersamling_mask', dtype=tf.complex64)
    temp = inputs_k_r
    phase = 96
    for j in range(1):
        conv1_k = complex_Conv1D(phase, 1, activation=None)(temp)
        conv1_k_d = Lambda(Concatenation_1d)([ temp,conv1_k])
        conv2_k = complex_Conv1D(phase, 1, activation=None)(conv1_k_d)
        conv2_k_d = Lambda(Concatenation_1d)([ temp,conv1_k, conv2_k])
        conv3_k = complex_Conv1D(phase, 1, activation='tanh')(conv2_k_d)
        conv3_k_d = Lambda(Concatenation_1d)([ temp,conv1_k, conv2_k, conv3_k])
        conv4_k = complex_Conv1D(phase, 1, activation=None)(conv3_k_d)
        conv4_k_d = Lambda(Concatenation_1d)([ temp,conv1_k, conv2_k, conv3_k, conv4_k])
        conv5_k = complex_Conv1D(phase, 1, activation=None)(conv4_k_d)
    k_net_out = add_dc_layer1(conv5_k, inputs, mask)
    temp1 = Lambda(iFourier2)(k_net_out)
    for i in range(15):

                conv1 = Mix(1,3,16,1,32)(temp1)
                conv2 = Mix(1,3,16,33,32)(conv1)
                conv3 = Mix(1,3,16,65,32)(conv2)
                conv4 = Mix(1,3,16,97,32)(conv3)
                conv5 = ComplexConv2D(kw=3, kh=3, n_out=1, sw=1, sh=1, activation=False)(conv4)
                block = Add()([(conv5), temp1])
                temp1 = add_dc_layer2(block, inputs, mask)


    out = temp1

    model = keras.Model(inputs=[inputs, mask], outputs=[out,k_net_out], name='deep_complex')
    return model


def generator_loss2(gene_output, gene_output_k,features, labels, masks):

    gene_l1l2_factor = 0.5
    l=1
    l1=0
    full_kspace = Lambda(Fourier2)(labels)
    gene_output_k_1 = tf.expand_dims(gene_output_k,-1)
    loss_kspace = tf.cast(tf.square(tf.abs(gene_output_k_1 - full_kspace)), tf.float32)
    # data consistency
    gene_dc_loss_1 = tf.reduce_mean(loss_kspace, name='gene_k_loss')
    gene_dc_loss_2 = tf.reduce_mean(tf.cast(tf.abs(gene_output_k_1 - full_kspace), tf.float32),name = 'gene_k_loss1')
    gene_dc_loss = gene_dc_loss_1 + gene_dc_loss_2

    # mse loss
    gene_l1_loss = tf.reduce_mean(tf.abs(gene_complex - labels), name='gene_l1_loss')
    gene_l2_loss = tf.reduce_mean(tf.square(tf.abs(gene_complex - labels)), name='gene_l2_loss')
    gene_mse_loss = tf.add( gene_l1_loss,  gene_l2_loss, name='gene_mse_loss')


    # gene_mse_factor as a parameter
    gene_loss = tf.add(gene_dc_loss, gene_mse_loss, name='gene_loss')
    return gene_loss
