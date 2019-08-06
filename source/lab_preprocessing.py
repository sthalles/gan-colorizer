import tensorflow as tf

def random_resize(image):
  # Random resize an image
  # For image of original size of 384x384
  # The output can have a maximum height and/or width of [461]
  # and minimum height and/or width of 307
  H, W = image.shape[:2]
  scale = tf.random.uniform([], minval=0.7, maxval=1.2, dtype=tf.float32, seed=None, name=None)
  shape = tf.stack((scale * W, scale * H), axis=0)
  shape = tf.cast(shape, tf.int32)
  image = tf.image.resize(image, size=shape)
  return image

def random_noise(input, target):
  bound = 1. / 128
  input += tf.random.uniform(shape=input.shape, minval=-bound, maxval=bound)
  return input, target

def process_tfds(features, HEIGHT, WIDTH):
  image = features["image"]
  image = tf.image.resize_with_crop_or_pad(image, target_height=HEIGHT, target_width=WIDTH)
  return tf.cast(image, tf.float32)

def random_crop(image, HEIGHT, WIDTH, CHANNELS=3):
  image = tf.image.random_crop(image, size=[HEIGHT, WIDTH, CHANNELS])
  return image

def random_flip(image):
  return tf.image.random_flip_left_right(image)