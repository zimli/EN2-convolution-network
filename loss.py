
import tensorflow as tf
from tensorflow import keras
from utils import ComplexInit
from tensorflow.keras.layers import Input, Lambda, Concatenate, Add, Multiply,Flatten
from tensorflow.keras import backend as K
from KIDENSE import *



def generator_loss2(gene_output_k, labels):

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
    gene_loss = tf.add( gene_l1_loss,  gene_l2_loss, name='gene_mse_loss')


    # gene_mse_factor as a parameter
    gene_loss = tf.add(gene_dc_loss, gene_loss, name='gene_loss')
    return gene_loss