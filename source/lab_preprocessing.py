import tensorflow as tf

def random_noise(input_, target):
  bound = 1. / 128
  input_ += tf.random.uniform(shape=input_.shape, minval=-bound, maxval=bound)
  target += tf.random.uniform(shape=target.shape, minval=-bound, maxval=bound)
  return input_, target

def process_tfds_train(features, HEIGHT, WIDTH):
  image = features["image"]
  image = tf.reshape(image, tf.shape(image))
#   image = tf.image.resize(image,
#         size=(288, 288), method=tf.image.ResizeMethod.BILINEAR,
#         preserve_aspect_ratio=True)
  image = tf.image.resize_with_crop_or_pad(image,target_height=HEIGHT+16,target_width=WIDTH+16)
  return tf.cast(image, tf.float32)

def process_tfds_test(features, HEIGHT, WIDTH):
  image = features["image"]
  image = tf.reshape(image, tf.shape(image))
  image = tf.image.resize_with_crop_or_pad(image, target_height=HEIGHT,target_width=WIDTH)
  return tf.cast(image, tf.float32)

def random_crop(image, HEIGHT, WIDTH, CHANNELS=3):
  image = tf.image.random_crop(image, size=[HEIGHT, WIDTH, CHANNELS])
  return image

def random_flip(image):
  return tf.image.random_flip_left_right(image)

def random_resize(image):
  # Random resize an image
  # For image of original size of 384x384
  # The output can have a maximum height and/or width of [461]
  # and minimum height and/or width of 307
  H, W = image.shape[:2]
  scale = tf.random.uniform([], minval=0.95, maxval=1.05, dtype=tf.float32, seed=None, name=None)
  shape = tf.stack((scale * W, scale * H), axis=0)
  shape = tf.cast(shape, tf.int32)
  image = tf.image.resize(image, size=shape)
  return image