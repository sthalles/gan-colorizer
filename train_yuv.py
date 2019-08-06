import tensorflow as tf
from discriminator.patch_resnet_discriminator import PatchResnetDiscriminator
from discriminator.resnet_discriminator import ResnetDiscriminator
from generator.unet_generator_64 import Generator
from generator.u_net_generator_v2 import UNetGenerator
from source.loss import disc_binary_cross_entropy, gen_l1_loss, gen_feature_matching, gen_binary_cross_entropy, disc_log_loss, \
  gen_log_loss, loss_hinge_dis, loss_hinge_gen
import time
import os
from source.preprocessing import *
from distutils.dir_util import copy_tree
import tensorflow_datasets as tfds
from shutil import copyfile

IMG_WIDTH = 384
IMG_HEIGHT = 384
PATCH_WIDTH = 128
PATCH_HEIGHT = 128
BATCH_SIZE = 64
BUFFER_SIZE = 2048
EPOCHS=20


# test_dataset = tf.data.Dataset.list_files("/home/thalles/Documents/valid_64x64/*.png")
test_dataset = tfds.load(name="coco2014", split=tfds.Split.VALIDATION)
test_dataset = test_dataset.map(lambda x: process_tfds(x, IMG_HEIGHT, IMG_WIDTH))
test_dataset = test_dataset.map(rgb2yuv, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.map(normalize_yuv)
test_dataset = test_dataset.batch(12)

train_dataset = tfds.load(name="coco2014", split=tfds.Split.ALL)
train_dataset = train_dataset.map(lambda x: process_tfds(x, IMG_HEIGHT, IMG_WIDTH))
train_dataset = train_dataset.map(random_resize)
train_dataset = train_dataset.map(lambda x: random_crop(x, PATCH_HEIGHT, PATCH_WIDTH))
train_dataset = train_dataset.map(random_flip)
train_dataset = train_dataset.map(rgb2yuv)
train_dataset = train_dataset.map(normalize_yuv)
train_dataset = train_dataset.map(random_noise)
train_dataset = train_dataset.repeat(1)
train_dataset = train_dataset.shuffle(32)
train_dataset = train_dataset.batch(BATCH_SIZE)

basefolder = os.path.join("records", str(time.time()))

generator = UNetGenerator(n_classes=3)
discriminator = PatchResnetDiscriminator(ch=64)

gen_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.9)
dis_optimizer = tf.keras.optimizers.Adam(4e-4, beta_1=0.0, beta_2=0.9)

summary_path = os.path.join(basefolder, 'summary')
train_summary_writer = tf.summary.create_file_writer(summary_path)

copy_tree('./generator', os.path.join(basefolder, 'generator'))
copy_tree('./discriminator', os.path.join(basefolder, 'discriminator'))
copyfile('./train_yuv.py', os.path.join(basefolder, 'train_yuv.py'))

retrain_from = '1562891802.672072'
if retrain_from is not None:
  checkpoint_dir = './records/' + retrain_from + '/checkpoints'
else:
  checkpoint_dir = os.path.join(basefolder, 'checkpoints')

checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                 discriminator_optimizer=dis_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

# load checkpoints to continue training
checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

def generate_images(model, test_input, target_images):
  # the training=True is intentional here since
  # we want the batch statistics while running the model
  # on the test dataset. If we use training=False, we will get
  # the accumulated statistics learned from the training dataset
  # (which we don't want)
  predictions = model(test_input, training=True)
  test_input = test_input.numpy()
  test_input = (test_input + 1) * 0.5

  predictions = predictions.numpy()
  predictions[..., 0] = (predictions[..., 0] + 1) * 0.5
  predictions = yuv2rgb(predictions)

  target_images = target_images.numpy()
  target_images[..., 0] = (target_images[..., 0] + 1) * 0.5
  target_images = yuv2rgb(target_images)

  tf.summary.image('generator_image', predictions, max_outputs=12, step=gen_optimizer.iterations)
  tf.summary.image('input_image', test_input, max_outputs=12, step=gen_optimizer.iterations)
  tf.summary.image('target_image', target_images, max_outputs=12, step=gen_optimizer.iterations)


kargs = {'training': True}


@tf.function
def train_step(input_image, target_image):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

    gen_output = generator(input_image, **kargs)

    patch_real = discriminator(input_image, target_image, **kargs)
    patch_fake = discriminator(input_image, gen_output, **kargs)

    gen_hinge_patch_loss = loss_hinge_gen(dis_fake=patch_fake)
    disc_hinge_patch_loss = loss_hinge_dis(dis_fake=patch_fake, dis_real=patch_real)

    l1_loss = gen_l1_loss(gen_output, target_image, lambda_=50)
    # gen_patch_loss = gen_binary_cross_entropy(disc_generated_output=patch_fake)
    # feature_matching = gen_feature_matching(features_real, features_fake)
    # disc_patch_loss = disc_binary_cross_entropy(disc_real_output=patch_real, disc_generated_output=patch_fake)

    # total loss
    gen_loss = gen_hinge_patch_loss + l1_loss
    disc_loss = disc_hinge_patch_loss

  # tf.summary.scalar('generator_patch_loss', gen_patch_loss, step=gen_optimizer.iterations)
  # tf.summary.scalar('generator_feature_matching', feature_matching, step=gen_optimizer.iterations)
  tf.summary.scalar('generator_l1_loss', l1_loss, step=gen_optimizer.iterations)
  tf.summary.scalar('generator_hinge_patch_loss', gen_hinge_patch_loss, step=gen_optimizer.iterations)
  tf.summary.scalar('generator_loss', gen_loss, step=gen_optimizer.iterations)

  # tf.summary.scalar('discriminator_hinge_loss', disc_hinge_loss, step=dis_optimizer.iterations)
  # tf.summary.scalar('discriminator_patch_loss', disc_patch_loss, step=dis_optimizer.iterations)
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

    for epoch in range(EPOCHS):

      for inp, tar in test_dataset.take(1):
        generate_images(generator, inp, tar)

      for input_image, target in dataset:
        train_step(input_image, target)

      # saving (checkpoint) the model every 20 epochs
      if epoch % 1 == 0:
        # generator.save(os.path.join(basefolder, 'generator'), save_format='tf')
        # discriminator.save(os.path.join(basefolder, 'discriminator'), save_format='tf')
        manager.save()
        print("New checkpoints saved.")
      print('Time taken for epoch {}\n'.format(epoch))


train(train_dataset)
