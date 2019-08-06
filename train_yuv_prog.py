import tensorflow as tf
from discriminator.patch_discriminator import Discriminator
from discriminator.patch_resnet_discriminator import PatchResnetDiscriminator
from discriminator.resnet_discriminator import ResnetDiscriminator
from generator.unet_generator_64 import Generator
from generator.ProgUNetGenerator import ProgUNetGenerator
from source.loss import disc_binary_cross_entropy, gen_l1_loss, gen_feature_matching, gen_binary_cross_entropy, disc_log_loss, \
  gen_log_loss, loss_hinge_dis, loss_hinge_gen
import time
import os
from source.preprocessing import *
from distutils.dir_util import copy_tree
import tensorflow_datasets as tfds
from shutil import copyfile

def generate_images(model, test_input, target_images):
  # the training=True is intentional here since
  # we want the batch statistics while running the model
  # on the test dataset. If we use training=False, we will get
  # the accumulated statistics learned from the training dataset
  # (which we don't want)
  predictions = model(test_input, training=True)
  test_input = test_input.numpy()
  test_input = (test_input + 1) * 0.5

  predictions = yuv2rgb(predictions)
  target_images = yuv2rgb(target_images)

  tf.summary.image('generator_image', predictions, max_outputs=12, step=gen_optimizer.iterations)
  tf.summary.image('input_image', test_input, max_outputs=12, step=gen_optimizer.iterations)
  tf.summary.image('target_image', target_images, max_outputs=12, step=gen_optimizer.iterations)


kargs = {'training': True}

PATCH_SIZES = [8,16,32,64,128,256]
BATCH_SIZES = [4,192,128,96,48,64,32]
EPOCHS =      [1,2,4,4,6,6,8]
IMG_SIZE = 384
BUFFER_SIZE = 8

folder_name=str(time.time())
basefolder = os.path.join("records", folder_name)

summary_path = os.path.join(basefolder, 'summary')
train_summary_writer = tf.summary.create_file_writer(summary_path)

copy_tree('./generator', os.path.join(basefolder, 'generator'))
copy_tree('./discriminator', os.path.join(basefolder, 'discriminator'))
copyfile('./train_yuv.py', os.path.join(basefolder, 'train_yuv.py'))

for PATCH_SIZE, BATCH_SIZE, EPOCH in zip(PATCH_SIZES,BATCH_SIZES,EPOCHS):
  print(PATCH_SIZE, BATCH_SIZE, EPOCH)
  restored = False
  # test_dataset = tf.data.Dataset.list_files("/home/thalles/Documents/valid_64x64/*.png")
  test_dataset = tfds.load(name="coco2014", split=tfds.Split.VALIDATION)
  test_dataset = test_dataset.map(lambda x: process_tfds(x, IMG_SIZE, IMG_SIZE))
  test_dataset = test_dataset.map(lambda x: random_crop(x, PATCH_SIZE, PATCH_SIZE))
  test_dataset = test_dataset.map(rgb2yuv, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  test_dataset = test_dataset.map(normalize_yuv)
  test_dataset = test_dataset.batch(12)

  train_dataset = tfds.load(name="coco2014", split=tfds.Split.VALIDATION)
  train_dataset = train_dataset.map(lambda x: process_tfds(x, IMG_SIZE, IMG_SIZE))
  train_dataset = train_dataset.map(random_resize)
  train_dataset = train_dataset.map(lambda x: random_crop(x, PATCH_SIZE, PATCH_SIZE))
  train_dataset = train_dataset.map(random_flip)
  train_dataset = train_dataset.map(rgb2yuv)
  train_dataset = train_dataset.map(normalize_yuv)
  train_dataset = train_dataset.map(random_noise)
  train_dataset = train_dataset.repeat(1)
  train_dataset = train_dataset.shuffle(1024)
  train_dataset = train_dataset.batch(BATCH_SIZE)

  generator = ProgUNetGenerator(n_classes=3)
  discriminator = ResnetDiscriminator(ch=64)

  gen_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.9)
  dis_optimizer = tf.keras.optimizers.Adam(4e-4, beta_1=0.0, beta_2=0.9)

  checkpoint_dir = './records/' + '1562805849.1027687' + '/checkpoints'
  checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                   discriminator_optimizer=dis_optimizer,
                                   generator=generator,
                                   discriminator=discriminator)
  manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=2)

  # load checkpoints to continue training
  checkpoint.restore(manager.latest_checkpoint).expect_partial()
  if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")


  # @tf.function
  def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      gen_output = generator(input_image, **kargs)
      # fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, constrained_layout=False)
      # ax1.imshow(tf.squeeze(input_image[0]) * 0.5 + 0.5, cmap='gray')
      # ax2.imshow(tf.squeeze(gen_output[0]) * 0.5 + 0.5)
      # ax3.imshow(target[0] * 0.5 + 0.5)
      # plt.show()

      disc_real_output, patch_real = discriminator(input_image, target, **kargs)
      disc_fake_output, patch_fake = discriminator(input_image, gen_output, **kargs)

      disc_hinge_loss = loss_hinge_dis(dis_fake=disc_fake_output, dis_real=disc_real_output)
      gen_hinge_loss = loss_hinge_gen(dis_fake=disc_fake_output)

      # gen_patch_loss = gen_binary_cross_entropy(disc_generated_output=patch_fake)
      # l1_loss = gen_l1_loss(gen_output, target)

      # disc_patch_loss = disc_binary_cross_entropy(disc_real_output=patch_real, disc_generated_output=patch_fake)

      # total loss
      gen_loss = gen_hinge_loss
      disc_loss = disc_hinge_loss

    # tf.summary.scalar('generator_patch_loss', gen_patch_loss, step=gen_optimizer.iterations)
    # tf.summary.scalar('generator_l1_loss', l1_loss, step=gen_optimizer.iterations)
    # tf.summary.scalar('generator_hinge_loss', gen_hinge_loss, step=gen_optimizer.iterations)
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


  with train_summary_writer.as_default():

    for epoch in range(EPOCH):

      # fake_input = tf.random.uniform(shape=(1, 4, 4, 1))
      # fake_output = generator(fake_input)
      # err = discriminator(fake_input,fake_output)
      # print(err)

      for input_image, target in train_dataset:
        train_step(input_image, target)

      for inp, tar in test_dataset.take(1):
        generate_images(generator, inp, tar)

      # saving (checkpoint) the model every 20 epochs
      if epoch % 1 == 0:
        # generator.save(os.path.join(basefolder, 'generator'), save_format='tf')
        # discriminator.save(os.path.join(basefolder, 'discriminator'), save_format='tf')
        manager.save()
        print("New checkpoints saved.")

      restored=True

      print('Time taken for epoch {}\n'.format(epoch))


