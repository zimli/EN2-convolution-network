import time
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#from utils_kdata import *
from KI_DENSE import *
import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
import os
import datetime
# Configs
print('[*] run basic configs ... ')
save_path = r''  #保存模型路径
save_path = os.path.join(save_path, '20230512-1757')
checkpoint_dir = os.path.join(save_path, "best_checkpoint")
testdata_path = r''  #测试集数据路径
BATCH_SIZE = 16
test_results_path = os.path.join(save_path, 'result')
try:
    os.mkdir(test_results_path)
except FileExistsError:
    pass

# Input Pipeline
print('[*] load data ... ')
labels_test, sparses_test,mask = data_process2(testdata_path)
test_dataset = tf.data.Dataset.from_tensor_slices((sparses_test[:128], labels_test[:128]))
test_dataset = test_dataset.batch(BATCH_SIZE)

mask = tf.cast(mask, tf.complex64)

# Build the Generator
print('[*] define model ... ')
slices, nw, nh, nz = sparses_test.shape
generator = getModel(nw, nh, nz)
slices = 128
# Define Optimizer
generator_optimizer = tf.keras.optimizers.Adam(0.5)

# Checkpoints (Object-based saving)
checkpoint = tf.train.Checkpoint(optimizer=generator_optimizer,
                                 model=generator)
ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.checkpoints[-1].split('-')[-1])
    checkpoint.restore(ckpt_manager.checkpoints[-1])
    print('[**] Latest checkpoint: {0} restored!'.format(start_epoch))
phase = 96
# Evaluate
steps = slices//BATCH_SIZE
cnn1 = np.zeros([steps, BATCH_SIZE, 96, phase, 2])
gt1 = np.zeros([steps, BATCH_SIZE, 96, phase, 1])
zf1 = np.zeros([steps, BATCH_SIZE, 96, phase, 2])
num = 0
step = 0
ZF_PATH, CNN_PATH, GT_PATH, CONCAT_PATH, Metrics_PATH = make_results_dir(test_results_path)

for inp, tar in test_dataset:
    step += 1
    mask2 = tf.expand_dims(mask, 0)
    mask2 = tf.tile(mask2, [inp.get_shape()[0], 1, 1, 1])
    mask2 = tf.reshape(mask2, mask2.get_shape()[:3])
    prediction,K = generator([inp, mask2])
    cnn1[step-1] = prediction
    gt1[step-1] = tf.abs(tar)
    zf1[step-1] = inp
    max_samples = tf.shape(inp)[0]
    image = tf.concat(axis=2, values=[tf.abs(tar), c2r(inp), c2r(prediction)])
    image = image[0:max_samples, :, :]
    image = tf.concat(axis=0, values=[image[i] for i in range(max_samples)])
    mpimg.imsave(os.path.join(CONCAT_PATH, '{:03d}.tif'.format(step)), tf.squeeze(image), cmap='gray')

    for i in range(max_samples):
        num += 1
        inp1=c2r(inp)
        prediction1=c2r(prediction)
        tar = tf.abs(tar)
        mpimg.imsave(os.path.join(ZF_PATH, '{:03d}.tif'.format(num)), inp1[i, :, :, 0], cmap='gray')
        mpimg.imsave(os.path.join(GT_PATH, '{:03d}.tif'.format(num)), tar[i, :, :, 0], cmap='gray')
        mpimg.imsave(os.path.join(CNN_PATH, '{:03d}.tif'.format(num)), prediction1[i, :, :, 0], cmap='gray')


cnn1 = np.reshape(cnn1, [-1, phase, phase, 2])
gt1 = np.reshape(gt1, [-1, phase, phase, 1])
zf1 = np.reshape(zf1, [-1, phase, phase, 2])
scipy.io.savemat(Metrics_PATH, {'cnn1': cnn1, 'gt1': gt1, 'zf1': zf1})

print("[**] Test Completed")

