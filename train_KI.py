import time
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import datetime
from utils import *
from KI_DENSE import *
import tensorflow as tf
tf.device('/gpu:2')
import matplotlib.pyplot as plt
from tensorflow import keras
# Configs
print('[*] run basic configs ... ')
save_path = r''   #保存模型路径
save_path = os.path.join(save_path, datetime.datetime.now().strftime("%Y%m%d-%H%M"))
checkpoint_dir = os.path.join(save_path, "checkpoint")
best_checkpoint_dir = os.path.join(save_path, "best_checkpoint")
traindata_path = r''  #数据路径

batchsize = 10
EPOCHS =100
SaveEpoch=10
BUFFER_SIZE = 1000
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Input Pipeline
print('[*] load data ... ')
labels, sparses, mask = data_process2(traindata_path)
idx = tf.random.shuffle(tf.range(labels.shape[0]))
train_dataset = tf.data.Dataset.from_tensor_slices((sparses[128:], labels[128:]))
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(batchsize)
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
# test
test_dataset = tf.data.Dataset.from_tensor_slices((sparses[:128], labels[:128]))
test_dataset = test_dataset.batch(batchsize)

mask = tf.cast(mask, tf.complex64)



# Build the Generator
print('[*] define model ... ')

nw, nh,nz = sparses.shape[1:]
generator = getModel(nw, nh,nz)
generator.summary()
train_loss = keras.metrics.Mean('train_loss', dtype=tf.float32)
test_loss = keras.metrics.Mean('test_loss', dtype=tf.float32)
exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.8)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=exponential_decay)

# Checkpoints (Object-based saving)
checkpoint = tf.train.Checkpoint(optimizer=generator_optimizer,
                                 model=generator)
best_checkpoint = tf.train.Checkpoint(optimizer=generator_optimizer,
                                 model=generator)
ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10)
best_ckpt_manager = tf.train.CheckpointManager(best_checkpoint, best_checkpoint_dir, max_to_keep=10)
start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    checkpoint.restore(ckpt_manager.latest_checkpoint)
    print('[**] Latest checkpoint restored!')

# Set up summary writers
train_log_dir = os.path.join(save_path, 'train')
test_log_dir = os.path.join(save_path, 'test')
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


# TRAINING
# Generate Images for visualization
def test_step2(model, test_input, mask, target):
    mask2 = tf.expand_dims(mask, 0)
    mask2 = tf.tile(mask2, [test_input.get_shape()[0], 1, 1, 1])
    mask2 = tf.reshape(mask2, mask2.get_shape()[:3])
    prediction,output_k = model([test_input, mask2], training=False)
    loss = generator_loss2(prediction, output_k,test_input, target, mask2)

    return loss


# Define Train Step: calculate the generator loss, calculate the gradients, and apply to optimizer
@tf.function
def train_step(input_image, mask, target):
    with tf.GradientTape() as gen_tape:
        mask1 = tf.expand_dims(mask,0)
        mask1 = tf.tile(mask1, [input_image.get_shape()[0], 1, 1, 1])
        mask1 = tf.reshape(mask1, mask1.get_shape()[:3])
        gen_output,gen_k = generator([input_image, mask1], training=True)
        gen_loss = generator_loss2(gen_output, gen_k,input_image, target, mask1)

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    return gen_loss


def train(dataset, epochs):
    train_loss_results = []
    test_loss_results = []
    best_test_loss=1e+11
    for epoch in range(start_epoch, epochs):
        print('[**] Start train Epoch {:03d}'.format(epoch+1))
        start = time.time()
        step = 0

        for input_image, target in dataset:
            gen_loss = train_step(input_image, mask, target)
            step += 1
            print("Epoch {:03d}: step: {:03d}, loss: {:.5f}".format(epoch+1, step, gen_loss))
            train_loss(gen_loss)

        # end epoch
        train_loss_results.append(train_loss.result())
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)

        for inp, tar in test_dataset:
            loss = test_step2(generator, inp, mask, tar)
            test_loss(loss)

        # end test
        test_loss_results.append(test_loss.result())
        if test_loss.result() < best_test_loss:
            best_ckpt_manager.save(checkpoint_number=epoch)
        best_test_loss = min(best_test_loss, test_loss.result())
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)

        plt.close('all')

        if (epoch + 1) % SaveEpoch == 0:
            ckpt_manager.save(checkpoint_number=epoch)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))

        # Reset metrics every epoch
        train_loss.reset_states()
        test_loss.reset_states()

    # End training
    print('[**] Training Completed')


if __name__ == '__main__':
    train(train_dataset, EPOCHS)

