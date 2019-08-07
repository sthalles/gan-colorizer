# import tensorflow as tf
# from skimage import color
# import numpy as np
#
# def rgb_to_gray(image):
#   return tf.image.rgb_to_grayscale(image), image
#
# def yuv2rgb(image):
#   image = tf.image.yuv_to_rgb(image)
#   image = tf.clip_by_value(image,clip_value_min=0, clip_value_max=1)
#   return image
#
# def random_resize(image):
#   H, W = image.shape[:2]
#   scale = tf.random.uniform([], minval=0.8, maxval=1.2, dtype=tf.float32, seed=None, name=None)
#   shape = tf.stack((scale * W, scale * H), axis=0)
#   shape = tf.cast(shape, tf.int32)
#   image = tf.image.resize(image, size=shape)
#   return image
#
# # normalizing the images to [-1, 1]
# def normalize(input_image, target_image):
#   input_image = (input_image / 127.5) - 1
#   target_image = (target_image / 127.5) - 1
#   return input_image, target_image
#
# def random_noise(input, target):
#   input += tf.random.uniform(shape=input.shape, minval=0., maxval=1. / 128)
#   return input, target
#
# def process_tfds(features, HEIGHT, WIDTH):
#   image = features["image"]
#   image = tf.image.resize_with_crop_or_pad(image, target_height=HEIGHT, target_width=WIDTH)
#   image = tf.cast(image, tf.float32)
#   return image
#
#
# def random_crop(image, HEIGHT, WIDTH, CHANNELS=3):
#   image = tf.image.random_crop(image, size=[HEIGHT, WIDTH, CHANNELS])
#   return image
#
# def random_flip(image):
#   return tf.image.random_flip_left_right(image)
#
# def rgb2yuv(image):
#   image =  tf.image.rgb_to_yuv(image/255)
#   return image
#
# def normalize_yuv(image):
#   return image[..., 0:1]  / 0.5 - 1, tf.concat([image[..., 0:1] / 0.5 - 1, image[..., 1:]], axis=-1)
#
#
# def LAB_normalizer(lab_image):
#   # receives batch input image in LAB color space
#   # Returns tuple (Luminance, AB channels)
#   batch_luminance = lab_image[..., 0:1]
#   batch_luminance = (batch_luminance / 50) - 1
#
#   batch_ab_channels = lab_image[..., 1:]
#   batch_ab_channels = batch_ab_channels / 128
#
#   batch_luminance = tf.cast(batch_luminance, tf.float32)
#   batch_ab_channels = tf.cast(batch_ab_channels, tf.float32)
#
#   return batch_luminance, batch_ab_channels
#
# def batch_LAB_denorm(L_batch, AB_batch):
#   # assert len(L_batch.shape) == 4, "L_batch image has to be 4D"
#   # assert len(AB_batch.shape) == 4, "AB_batch image has to be 4D"
#
#   L_batch = (L_batch + 1) * 50
#   AB_batch = AB_batch * 128
#   LAB_image = tf.concat([L_batch, AB_batch], axis=-1)
#   return LAB_image
#
# def batch_LAB2RGB(LAB_batch):
#   # assert len(LAB_batch.shape) == 4, "LAB_batch image has to be 4D"
#   RGB_batch = np.zeros(LAB_batch.shape)
#   for i in range(LAB_batch.shape[0]):
#     RGB_batch[i] = color.lab2rgb(LAB_batch[i])
#   return RGB_batch