import tensorflow as tf
from tensorflow import keras
from utils import ComplexInit
from tensorflow.keras.layers import Input, Lambda, Concatenate, Add, Multiply,Flatten
from tensorflow.keras import backend as K
from EN2 import EN2
from FMU import ComplexConv2D, FMU

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


def getModel(img_width, img_height, channels):
    inputs = Input(shape=[img_width, img_height, channels], name='zero-filling')  # 96*96*1
    inputs_k_c = Lambda(Fourier2)(inputs)
    inputs_k_r = Lambda(c2r1)(inputs_k_c)
    mask = Input(shape=[img_width, img_height], name='undersamling_mask', dtype=tf.complex64)
    temp = inputs_k_r
    phase = 96
    for j in range(1):
        conv1_k = EN2(phase, 1, activation=None)(temp)
        conv2_k = EN2(phase, 1, activation=None)(conv1_k)
        conv3_k = EN2(phase, 1, activation='tanh')(conv2_k)
        conv4_k = EN2(phase, 1, activation=None)(conv3_k)
        conv5_k = EN2(phase, 1, activation=None)(conv4_k)
    k_net_out = add_dc_layer1(conv5_k, inputs, mask)
    temp1 = Lambda(iFourier2)(k_net_out)
    for i in range(15):

                conv1 = FMU(1,3,16,1,32)(temp1)
                conv2 = FMU(1,3,16,33,32)(conv1)
                conv3 = FMU(1,3,16,65,32)(conv2)
                conv4 = FMU(1,3,16,97,32)(conv3)
                conv5 = ComplexConv2D(kw=3, kh=3, n_out=1, sw=1, sh=1, activation=False)(conv4)
                block = Add()([(conv5), temp1])
                temp1 = add_dc_layer2(block, inputs, mask)


    out = temp1

    model = keras.Model(inputs=[inputs, mask], outputs=[out,k_net_out], name='deep_complex')
    return model



