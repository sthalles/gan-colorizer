import tensorflow as tf
# from discriminator.patch_resnet_discriminator import PatchResnetDiscriminator
from discriminator.discriminator import Discriminator
from generator.generator import UNetGenerator
from source.loss import gen_l1_loss, loss_hinge_dis, loss_hinge_gen, gen_binary_cross_entropy, disc_binary_cross_entropy
import time
import os
# from source.preprocessing import *
from distutils.dir_util import copy_tree
import tensorflow_datasets as tfds
from shutil import copyfile
from source.lab_preprocessing import *
from source.image_processing import *
import yaml
import importlib
import numpy as np
import matplotlib.pyplot as plt

# Read YAML file
with open("./config/coco.yml", 'r') as stream:
    meta_parameters = yaml.safe_load(stream)

DISC_UPDATE = meta_parameters['disc_update']
IMG_SIZE = meta_parameters['dataset']['img_size']
BATCH_SIZE = meta_parameters['batchsize']
BUFFER_SIZE = 2048
EPOCHS = meta_parameters['epochs']
SUMMARY_EVERY_N_STEPS = meta_parameters['summary_every_n_steps']
SAVE_EVERY_N_STEPS = meta_parameters['save_every_n_steps']
PATCH_SIZE = meta_parameters['dataset']['img_size']
DATASET_NAME = meta_parameters['dataset']['name']

# test_dataset = tf.data.Dataset.list_files("/home/thalles/Documents/valid_64x64/*.png")
# test_dataset = tfds.load(name=DATASET_NAME, split=tfds.Split.TEST)
# test_dataset = test_dataset.map(lambda x: process_tfds_test(x, IMG_SIZE, IMG_SIZE))
# test_dataset = test_dataset.map(rgb_to_lab)
# test_dataset = test_dataset.map(preprocess_lab)
# test_dataset = test_dataset.repeat(1)
# test_dataset = test_dataset.batch(12)

train_dataset = tfds.load(name=DATASET_NAME, split=tfds.Split.ALL)
train_dataset = train_dataset.map(lambda x: process_tfds_train(x, IMG_SIZE, IMG_SIZE))
train_dataset = train_dataset.map(random_resize)
train_dataset = train_dataset.map(lambda x: random_crop(x, PATCH_SIZE, PATCH_SIZE))
train_dataset = train_dataset.map(random_flip)
train_dataset = train_dataset.map(rgb_to_lab)
train_dataset = train_dataset.map(preprocess_lab)
# train_dataset = train_dataset.map(random_noise)
train_dataset = train_dataset.repeat(EPOCHS)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

gen_parameters = meta_parameters['model']['generator']
disc_parameters = meta_parameters['model']['discriminator']

basefolder = os.path.join("records", str(time.time()))
generator_lib = importlib.import_module(gen_parameters['name'])
discriminator_lib = importlib.import_module(disc_parameters['name'])

generator = generator_lib.UNetGenerator(**gen_parameters['args'])
discriminator = discriminator_lib.Discriminator(**disc_parameters['args'])

gen_optimizer_args = meta_parameters['optimizer']['generator']
gen_optimizer = tf.keras.optimizers.Adam(**gen_optimizer_args)

dis_optimizer_args = meta_parameters['optimizer']['discriminator']
dis_optimizer = tf.keras.optimizers.Adam(**dis_optimizer_args)

summary_path = os.path.join(basefolder, 'summary')
train_summary_writer = tf.summary.create_file_writer(summary_path)

copy_tree('./generator', os.path.join(basefolder, 'generator'))
copy_tree('./discriminator', os.path.join(basefolder, 'discriminator'))
copyfile('./train_lab.py', os.path.join(basefolder, 'train_lab.py'))

retrain_from = None
if retrain_from is not None:
    checkpoint_dir = './records/' + retrain_from + '/checkpoints'
else:
    checkpoint_dir = os.path.join(basefolder, 'checkpoints')

checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                 discriminator_optimizer=dis_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
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

    fake_image = model(L_batch, sn_update=False, training=True)

    a_chan_fake, b_chan_fake = tf.unstack(fake_image, axis=3)
    fake_lab_image = deprocess_lab(L_batch, a_chan_fake, b_chan_fake)
    fake_rgb_image = lab_to_rgb(fake_lab_image)

    a_chan, b_chan = tf.unstack(AB_batch, axis=3)
    real_lab_image = deprocess_lab(L_batch, a_chan, b_chan)
    real_rgb_image = lab_to_rgb(real_lab_image)

    tf.summary.image('generator_image', fake_rgb_image, max_outputs=12, step=gen_optimizer.iterations)
    tf.summary.image('input_image', (L_batch + 1) * 0.5, max_outputs=12, step=gen_optimizer.iterations)
    tf.summary.image('target_image', real_rgb_image, max_outputs=12, step=gen_optimizer.iterations)


kwargs = {'training': True}


# # @tf.function
# def generator_train_step(L_batch, AB_batch):
#   with tf.GradientTape() as gen_tape:
#
#     fake_image = generator(L_batch, sn_update=True, **kargs)
#     disc_fake_loss = discriminator(tf.concat([L_batch, fake_image], axis=3), sn_update=True)
#
#     regularization_loss = tf.math.add_n(generator.losses)
#     hinge_loss = loss_hinge_gen(dis_fake=disc_fake_loss)
#
#     # L1 Loss could compare the 3 channel image instead of only the AB channel
#     l1_loss = gen_l1_loss(fake_image, AB_batch, lambda_=100)
#     gen_loss = hinge_loss + regularization_loss + l1_loss
#
#   tf.summary.scalar('generator_l1_loss', l1_loss, step=gen_optimizer.iterations)
#   tf.summary.scalar('generator_hinge_loss', hinge_loss, step=gen_optimizer.iterations)
#   tf.summary.scalar('generator_regularization_loss', regularization_loss, step=gen_optimizer.iterations)
#   tf.summary.scalar('generator_total_loss', gen_loss, step=gen_optimizer.iterations)
#
#   generator_gradients = gen_tape.gradient(gen_loss,
#                                           generator.trainable_variables)
#   gen_optimizer.apply_gradients(zip(generator_gradients,
#                                     generator.trainable_variables))
#
# # @tf.function
# def discriminator_train_step(L_batch, AB_batch):
#   with tf.GradientTape() as disc_tape:
#     disc_real_loss = discriminator(tf.concat([L_batch, AB_batch], axis=3), sn_update=True)
#
#     fake_batch = generator(L_batch, sn_update=True, **kargs)
#
#     disc_fake_loss = discriminator(tf.concat([L_batch, fake_batch], axis=3), sn_update=True)
#
#     disc_loss = loss_hinge_dis(dis_fake=disc_fake_loss, dis_real=disc_real_loss)
#
#   tf.summary.scalar('discriminator_loss', disc_loss, step=dis_optimizer.iterations)
#   discriminator_gradients = disc_tape.gradient(disc_loss,
#                                                discriminator.trainable_variables)
#
#   dis_optimizer.apply_gradients(zip(discriminator_gradients,
#                                     discriminator.trainable_variables))

# @tf.function
def train_step(L_batch, AB_batch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_batch = generator(L_batch, sn_update=True, **kwargs)

        disc_patch_real = discriminator(tf.concat([L_batch, AB_batch], axis=3), sn_update=True, **kwargs)
        disc_patch_fake = discriminator(tf.concat([L_batch, fake_batch], axis=3), sn_update=True, **kwargs)

        gen_patch_loss = gen_binary_cross_entropy(disc_generated_output=disc_patch_fake)
        disc_loss = disc_binary_cross_entropy(disc_real_output=disc_patch_real, disc_generated_output=disc_patch_fake)

        # L1 Loss could compare the 3 channel image instead of only the AB channel
        l1_loss = gen_l1_loss(fake_batch, AB_batch, lambda_=100)

        # total generator loss
        gen_loss = gen_patch_loss + l1_loss

    tf.summary.histogram(name="real_image_discributions", data=AB_batch, step=dis_optimizer.iterations)
    tf.summary.histogram(name="fake_image_discributions", data=fake_batch, step=dis_optimizer.iterations)

    tf.summary.scalar('generator_l1_loss', l1_loss, step=gen_optimizer.iterations)
    tf.summary.scalar('generator_patch_loss', gen_patch_loss, step=gen_optimizer.iterations)
    tf.summary.scalar('generator_loss', gen_loss, step=gen_optimizer.iterations)
    tf.summary.scalar('discriminator_loss', disc_loss, step=dis_optimizer.iterations)

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

        for L_batch, AB_batch in train_dataset:

            if tf.math.equal(gen_optimizer.iterations % SUMMARY_EVERY_N_STEPS, 0):
                generate_images(generator, L_batch, AB_batch)

            # if tf.math.equal(dis_optimizer.iterations % DISC_UPDATE, 0):
            #   # perform a generator update
            #   generator_train_step(L_batch, AB_batch)
            #
            # discriminator_train_step(L_batch, AB_batch)
            train_step(L_batch, AB_batch)

        manager.save()
        print("New checkpoints saved.")


train()
