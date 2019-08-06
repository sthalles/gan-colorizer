import tensorflow as tf
from discriminator.patch_resnet_discriminator import PatchResnetDiscriminator
from generator.u_net_generator_v2 import UNetGenerator
from source.loss import gen_l1_loss, loss_hinge_dis, loss_hinge_gen
import time
import os
# from source.preprocessing import *
from distutils.dir_util import copy_tree
import tensorflow_datasets as tfds
from shutil import copyfile
from source.lab_preprocessing import *
from source.image_processing import *

IMG_SIZE = 384
PATCH_SIZE = 128
BATCH_SIZE = 48
BUFFER_SIZE = 2048
EPOCHS = 4

IMG_SHAPE = (PATCH_SIZE, PATCH_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = True

# test_dataset = tf.data.Dataset.list_files("/home/thalles/Documents/valid_64x64/*.png")
test_dataset = tfds.load(name="coco2014", split=tfds.Split.VALIDATION)
test_dataset = test_dataset.map(lambda x: process_tfds(x, IMG_SIZE, IMG_SIZE))
test_dataset = test_dataset.map(rgb_to_lab)
test_dataset = test_dataset.map(preprocess_lab)
test_dataset = test_dataset.repeat(1)
test_dataset = test_dataset.batch(12)

train_dataset = tfds.load(name="coco2014", split=tfds.Split.ALL)
train_dataset = train_dataset.map(lambda x: process_tfds(x, IMG_SIZE, IMG_SIZE))
train_dataset = train_dataset.map(random_resize)
train_dataset = train_dataset.map(lambda x: random_crop(x, PATCH_SIZE, PATCH_SIZE))
train_dataset = train_dataset.map(random_flip)
train_dataset = train_dataset.map(rgb_to_lab)
train_dataset = train_dataset.map(preprocess_lab)
train_dataset = train_dataset.repeat(1)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

basefolder = os.path.join("records", str(time.time()))

generator = UNetGenerator(n_classes=2)
# discriminator = PatchResnetDiscriminator(ch=64)

gen_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.9)
dis_optimizer = tf.keras.optimizers.Adam(4e-4, beta_1=0.0, beta_2=0.9)

summary_path = os.path.join(basefolder, 'summary')
train_summary_writer = tf.summary.create_file_writer(summary_path)

copy_tree('./generator', os.path.join(basefolder, 'generator'))
copy_tree('./discriminator', os.path.join(basefolder, 'discriminator'))
copyfile('./train_lab.py', os.path.join(basefolder, 'train_yuv.py'))

retrain_from = '1562972964.7331862'
if retrain_from is not None:
  checkpoint_dir = './records/' + retrain_from + '/checkpoints'
else:
  checkpoint_dir = os.path.join(basefolder, 'checkpoints')

checkpoint = tf.train.Checkpoint(generator=generator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=2)

# load checkpoints to continue training
checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
  print("Restored from {}".format(manager.latest_checkpoint))
else:
  print("Initializing from scratch.")


def generate_images(model, L_batch, AB_batch):
  # the training=True is intentional here since
  # we want the batch statistics while running the model
  # on the test dataset. If we use training=False, we will get
  # the accumulated statistics learned from the training dataset
  # (which we don't want)

  fake_image = model(L_batch, training=True)

  a_chan_fake, b_chan_fake = tf.unstack(fake_image, axis=3)
  fake_lab_image = deprocess_lab(L_batch, a_chan_fake, b_chan_fake)
  fake_rgb_image = lab_to_rgb(fake_lab_image)

  a_chan, b_chan = tf.unstack(AB_batch, axis=3)
  real_lab_image = deprocess_lab(L_batch, a_chan, b_chan)
  real_rgb_image = lab_to_rgb(real_lab_image)

  tf.summary.image('generator_image', fake_rgb_image, max_outputs=12, step=gen_optimizer.iterations)
  tf.summary.image('input_image', (L_batch + 1) * 0.5, max_outputs=12, step=gen_optimizer.iterations)
  tf.summary.image('target_image', real_rgb_image, max_outputs=12, step=gen_optimizer.iterations)


kargs = {'training': True}

layer_name = 'Conv_1'
discriminator = tf.keras.Model(inputs=base_model.input,
                                 outputs=base_model.get_layer(layer_name).output)

@tf.function
def train_step(L_batch, AB_batch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(L_batch, **kargs)

    # reconstruct RGB original image
    a_chan_fake, b_chan_fake = tf.unstack(AB_batch, axis=3)
    real_lab_image = deprocess_lab(L_batch, a_chan_fake, b_chan_fake)
    real_rgb_image = lab_to_rgb(real_lab_image)

    # reconstruct RGB fake image
    a_chan_fake, b_chan_fake = tf.unstack(gen_output, axis=3)
    fake_lab_image = deprocess_lab(L_batch, a_chan_fake, b_chan_fake)
    fake_rgb_image = lab_to_rgb(fake_lab_image)

    patch_real = discriminator(real_rgb_image, **kargs)
    patch_fake = discriminator(fake_rgb_image, **kargs)

    gen_hinge_patch_loss = loss_hinge_gen(dis_fake=patch_fake)
    disc_loss = loss_hinge_dis(dis_fake=patch_fake, dis_real=patch_real)

    # perceptural_loss_real = gen_l1_loss(imagenet_features, features_real, lambda_=5)
    # perceptural_loss_fake = gen_l1_loss(imagenet_features, features_fake, lambda_=5)

    # perceptual loss
    # perceptural_loss = gen_l1_loss(imagenet_features, mean_features, lambda_=3.5)

    # L1 Loss could compare the 3 channel image instead of only the AB channel
    l1_loss = gen_l1_loss(gen_output, AB_batch, lambda_=35)

    # total generator loss
    gen_loss = gen_hinge_patch_loss + l1_loss

  # tf.summary.scalar('generator_patch_loss', gen_patch_loss, step=gen_optimizer.iterations)
  # tf.summary.scalar('generator_perceptural_loss_real', perceptural_loss_real, step=gen_optimizer.iterations)
  # tf.summary.scalar('generator_perceptural_loss_fake', perceptural_loss_fake, step=gen_optimizer.iterations)
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


def train():
  with train_summary_writer.as_default():

    for epoch in range(EPOCHS):

      for L_batch, AB_batch in test_dataset.take(1):
        generate_images(generator, L_batch, AB_batch)

      for L_batch, AB_batch in train_dataset:
        train_step(L_batch, AB_batch)

      # generator.save(os.path.join(basefolder, 'generator'), save_format='tf')
      # discriminator.save(os.path.join(basefolder, 'discriminator'), save_format='tf')
      manager.save()
      print("New checkpoints saved.")
      print('Time taken for epoch {}\n'.format(epoch))

train()
