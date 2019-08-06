import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from discriminator.patch_resnet_discriminator import PatchResnetDiscriminator
from generator.u_net_generator_v2 import UNetGenerator
from collections import Counter
from source.preprocessing import rgb_to_gray, normalize
import tensorflow_datasets as tfds
from source.image_processing import *
from source.lab_preprocessing import *

IMG_WIDTH = 384
IMG_HEIGHT = 384
IMG_SIZE = IMG_HEIGHT
BATCH_SIZE = 3


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    image = tf.reshape(image, (IMG_HEIGHT, IMG_WIDTH, 3))
    return image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    # input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


# test_dataset = tf.data.Dataset.list_files("/home/thalles/Documents/*.jpg")
# # test_dataset = tfds.load(name="coco2014", split=tfds.Split.TEST)
# test_dataset = test_dataset.map(load)
# test_dataset = test_dataset.map(rgb_to_gray, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# test_dataset = test_dataset.map(normalize)
# test_dataset = test_dataset.shuffle(32)
# test_dataset = test_dataset.batch(BATCH_SIZE)


# test_dataset = tf.data.Dataset.list_files("/home/thalles/Documents/valid_64x64/*.png")
test_dataset = tfds.load(name="imagenet2012", split=tfds.Split.ALL)
test_dataset = test_dataset.map(lambda x: process_tfds(x, IMG_SIZE, IMG_SIZE))
test_dataset = test_dataset.map(rgb_to_lab)
test_dataset = test_dataset.map(preprocess_lab)
test_dataset = test_dataset.repeat(1)
test_dataset = test_dataset.batch(BATCH_SIZE)

# test_dataset = tf.data.Dataset.list_files("/home/thalles/Documents/*.jpg")
# # shuffling so that for every epoch a different image is generated
# # to predict and display the progress of our model.
# test_dataset = test_dataset.map(load_image_test)
# test_dataset = test_dataset.map(rgb_to_gray, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# test_dataset = test_dataset.batch(BATCH_SIZE)

basefolder = "./records/1562972964.7331862"

generator = UNetGenerator(n_classes=2)
discriminator = PatchResnetDiscriminator(ch=64)

gen_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.9)
dis_optimizer = tf.keras.optimizers.Adam(4e-4, beta_1=0.0, beta_2=0.9)

summary_path = os.path.join(basefolder, 'summary')
train_summary_writer = tf.summary.create_file_writer(summary_path)

checkpoint_dir = os.path.join(basefolder, 'checkpoints')
checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                 discriminator_optimizer=dis_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")


def generate_images(model, test_input, target):
    # the training=True is intentional here since
    # we want the batch statistics while running the model
    # on the test dataset. If we use training=False, we will get
    # the accumulated statistics learned from the training dataset
    # (which we don't want)
    print(test_input.shape)
    prediction = model(test_input, training=True)
    return prediction

def generate_images(model, L_batch, AB_batch):
  # the training=True is intentional here since
  # we want the batch statistics while running the model
  # on the test dataset. If we use training=False, we will get
  # the accumulated statistics learned from the training dataset
  # (which we don't want)
  print(AB_batch.shape)
  fake_image = model(L_batch, training=True)

  a_chan, b_chan = tf.unstack(AB_batch, axis=3)
  real_lab_image = deprocess_lab(L_batch, a_chan, b_chan)
  real_rgb_image = lab_to_rgb(real_lab_image)

  a_chan_fake, b_chan_fake = tf.unstack(fake_image, axis=3)
  fake_lab_image = deprocess_lab(L_batch, a_chan_fake, b_chan_fake)
  fake_rgb_image = lab_to_rgb(fake_lab_image)

  return fake_rgb_image, real_rgb_image


nrow = 2
ncol = BATCH_SIZE


def get_entropy(image):
    image = image.flatten().astype(np.int32)
    counts = list(Counter(image).values())
    probs = np.array(counts) / image.shape[0]
    E = -np.sum(probs * np.log2(probs))
    return E


entropy_absolute_error = []
for inp, tar in test_dataset.take(20):
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, constrained_layout=False)
    gen_images, real_rgb_image = generate_images(generator, inp, tar)
    print("Input range:", np.min(inp), np.max(inp), "shape:", inp.shape)
    print("Target range:", np.min(tar), np.max(tar), "shape:", tar.shape)
    print("Gen Image range:", np.min(gen_images), np.max(gen_images), "shape:", gen_images.shape)

    for i in range(ncol):
        # calculate entropy
        im1 = (tar[i].numpy().squeeze() * 0.5 + 0.5) * 255
        im2 = (gen_images[i].numpy().squeeze() * 0.5 + 0.5) * 255

        E = get_entropy(im1)
        E_hat = get_entropy(im2)
        EAD = np.abs(E - E_hat)
        entropy_absolute_error.append(EAD)
        print("Entropy (train):", E, "\tEntropy (generated):", E_hat, "\tDifference:", EAD)

        # axs[0][i].imshow(tf.squeeze(inp[i]) * 0.5 + 0.5, cmap="gray")
        axs[0][i].imshow(tf.squeeze(gen_images[i]))
        axs[0][i].set_title("Difference: {:.02f} ".format(EAD))
        axs[1][i].imshow(tf.squeeze(real_rgb_image[i]))
    plt.show()

print("Entropy error:", np.array(entropy_absolute_error).mean())