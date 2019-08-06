import tensorflow as tf
from discriminator.discriminator_64 import Discriminator
from generator.unet_generator_64 import Generator
from source.loss import gen_supervised_loss, disc_binary_cross_entropy
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from source.preprocessing import *
from distutils.dir_util import copy_tree
from skimage.transform import resize
import tensorflow_datasets as tfds

IMG_WIDTH = 384
IMG_HEIGHT = 384
PATCH_WIDTH=128
PATCH_HEIGHT=128
BATCH_SIZE = 64
BUFFER_SIZE = 2048


def process_tfds(features):
  image = features["image"]
  image = tf.image.resize_with_crop_or_pad(image, target_height=IMG_HEIGHT, target_width=IMG_WIDTH)
  image = tf.cast(image, tf.float32)
  return image

def random_crop(image):
  image = tf.image.random_crop(image, size=[PATCH_HEIGHT, PATCH_WIDTH, 3])
  return image

# test_dataset = tf.data.Dataset.list_files("/home/thalles/Documents/valid_64x64/*.png")
test_dataset = tfds.load(name="coco2014", split=tfds.Split.VALIDATION)
test_dataset = test_dataset.map(process_tfds)
test_dataset = test_dataset.map(rgb_to_gray, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.map(normalize)
test_dataset = test_dataset.batch(12)

train_dataset = tfds.load(name="coco2014", split=tfds.Split.ALL)
train_dataset = train_dataset.map(process_tfds)
train_dataset = train_dataset.map(random_resize)
train_dataset = train_dataset.map(random_crop)
train_dataset = train_dataset.map(rgb_to_gray, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.repeat(1)
train_dataset = train_dataset.shuffle(1024)
train_dataset = train_dataset.batch(BATCH_SIZE)

basefolder = os.path.join("records", str(time.time()))

generator = Generator(n_classes=3, in_depth=1, activation='lrelu')
discriminator = Discriminator(n_classes=3, in_depth=1, activation='lrelu')

gen_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.9)
dis_optimizer = tf.keras.optimizers.Adam(4e-4, beta_1=0.0, beta_2=0.9)

summary_path = os.path.join(basefolder, 'summary')
train_summary_writer = tf.summary.create_file_writer(summary_path)
checkpoint_dir = os.path.join(basefolder, 'checkpoints')

copy_tree('./generator', os.path.join(basefolder, 'generator'))
copy_tree('./discriminator', os.path.join(basefolder, 'discriminator'))

checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                 discriminator_optimizer=dis_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)


def generate_images(model, test_input, target):
  # the training=True is intentional here since
  # we want the batch statistics while running the model
  # on the test dataset. If we use training=False, we will get
  # the accumulated statistics learned from the training dataset
  # (which we don't want)
  prediction = model(test_input, training=True)
  tf.summary.image('generator_image', prediction * 0.5 + 0.5, max_outputs=12, step=gen_optimizer.iterations)
  tf.summary.image('input_image', test_input * 0.5 + 0.5, max_outputs=12, step=gen_optimizer.iterations)
  tf.summary.image('target_image', target * 0.5 + 0.5, max_outputs=12, step=gen_optimizer.iterations)


@tf.function
def train_step(input_image, target):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

    gen_output = generator(input_image, training=True)
    print("Input range:", np.min(input_image), np.max(input_image), "shape:", input_image.shape)
    print("Target range:", np.min(target), np.max(target), "shape:", target.shape)
    print("Gen Image range:", np.min(gen_output), np.max(gen_output), "shape:", gen_output.shape)
    # fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, constrained_layout=False)
    # ax1.imshow(tf.squeeze(input_image[0]) * 0.5 + 0.5, cmap='gray')
    # ax2.imshow(tf.squeeze(gen_output[0]) * 0.5 + 0.5)
    # ax3.imshow(target[0] * 0.5 + 0.5)
    # plt.show()

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_loss = gen_supervised_loss(disc_generated_output, gen_output, target)
    disc_loss = disc_binary_cross_entropy(disc_real_output, disc_generated_output)

  tf.summary.scalar('generator_loss', gen_loss, step=gen_optimizer.iterations)
  tf.summary.scalar('discriminator_loss', disc_loss, step=dis_optimizer.iterations)
  # tf.summary.image('input_image', input_image * 0.5 + 0.5, max_outputs=12, step=gen_optimizer.iterations)
  # tf.summary.image('generator_image', gen_output * 0.5 + 0.5, max_outputs=12, step=gen_optimizer.iterations)

  generator_gradients = gen_tape.gradient(gen_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  gen_optimizer.apply_gradients(zip(generator_gradients,
                                    generator.trainable_variables))
  dis_optimizer.apply_gradients(zip(discriminator_gradients,
                                    discriminator.trainable_variables))


def train(dataset):
  with train_summary_writer.as_default():
    epoch = 1
    while True:

      for inp, tar in test_dataset.take(1):
        generate_images(generator, inp, tar)

      for input_image, target in dataset:
        train_step(input_image, target)

      # saving (checkpoint) the model every 20 epochs
      if epoch % 2 == 0:
        generator.save(os.path.join(basefolder, 'generator'), save_format='tf')
        discriminator.save(os.path.join(basefolder, 'discriminator'), save_format='tf')
        manager.save()
        print("New checkpoints saved.")

      print('Time taken for epoch {}\n'.format(epoch))
      epoch += 1


train(train_dataset)
