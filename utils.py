from tensorflow.python.ops.init_ops import Initializer,_compute_fans
from numpy.random import RandomState
import numpy as np
import scipy.io
import tensorflow as tf
import os


class ComplexInit(Initializer):

    def __init__(self, kernel_size, input_dim,
                 weight_dim, nb_filters=None,
                 criterion='glorot', seed=None):

        # `weight_dim` is used as a parameter for sanity check
        # as we should not pass an integer as kernel_size when
        # the weight dimension is >= 2.
        # nb_filters == 0 if weights are not convolutional (matrix instead of filters)
        # then in such a case, weight_dim = 2.
        # (in case of 2D input):
        #     nb_filters == None and len(kernel_size) == 2 and_weight_dim == 2
        # conv1D: len(kernel_size) == 1 and weight_dim == 1
        # conv2D: len(kernel_size) == 2 and weight_dim == 2
        # conv3d: len(kernel_size) == 3 and weight_dim == 3

        assert len(kernel_size) == weight_dim and weight_dim in {0, 1, 2, 3}
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.criterion = criterion
        self.seed = 1337 if seed is None else seed

    def __call__(self, shape, dtype=None, partition_info=None):

        if self.nb_filters is not None:
            kernel_shape = tuple(self.kernel_size) + (int(self.input_dim), self.nb_filters)
        else:
            kernel_shape = (int(self.input_dim), self.kernel_size[-1])

        fan_in, fan_out = _compute_fans(
            tuple(self.kernel_size) + (self.input_dim, self.nb_filters)
        )

        if self.criterion == 'glorot':
            s = 1. / (fan_in + fan_out)
        elif self.criterion == 'he':
            s = 1. / fan_in
        else:
            raise ValueError('Invalid criterion: ' + self.criterion)
        rng = RandomState(self.seed)
        modulus = rng.rayleigh(scale=s, size=kernel_shape)
        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
        weight_real = modulus * np.cos(phase)
        weight_imag = modulus * np.sin(phase)
        weight = np.concatenate([weight_real, weight_imag], axis=-1)

        return weight


def data_process2(data_path):
    """
    process data for train or test
    :param data_path: Train or test data path
    :return: labels, sparses, mask
    """
    X = scipy.io.loadmat(data_path)
    labels = X['labels'].astype(np.float32)
    labels = np.rollaxis(labels, 3, 0)
    labels = labels[..., np.newaxis]

    sparses = X['sparses'].astype(np.float32)
    sparses = np.rollaxis(sparses, 3, 0)

    mask = X['mask'].astype(np.float32)
    #mask_h = X['mask_h'].astype(np.float32)
    mask = mask[:, :, np.newaxis]
    #mask_h = mask_h[:, :, np.newaxis]



    return labels, sparses,mask


def c2r(x):
    # convert a two channel complex image to one channel real image
    x_complex = tf.complex(x[..., 0], x[..., 1])
    x_mag = tf.abs(x_complex)
    x_mag = tf.expand_dims(x_mag, -1)
    return x_mag


def make_results_dir(test_results_path):
    """
    Make dir for test results save
    :param test_results_path:
    :return: ZF, CNN, GT
    """
    ZF_PATH = os.path.join(test_results_path, 'ZF')
    CNN_PATH = os.path.join(test_results_path, 'CNN')
    GT_PATH = os.path.join(test_results_path, 'GT')
    CONCAT_PATH = os.path.join(test_results_path, 'CONCAT')
    Metrics_PATH = os.path.join(test_results_path, 'metrics.mat')
    try:
        os.mkdir(ZF_PATH)
        os.mkdir(CNN_PATH)
        os.mkdir(GT_PATH)
        os.mkdir(CONCAT_PATH)
    except FileExistsError:
        pass
    return ZF_PATH, CNN_PATH, GT_PATH, CONCAT_PATH, Metrics_PATH

