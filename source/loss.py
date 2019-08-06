import tensorflow as tf

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
  loss = tf.reduce_mean(tf.nn.relu(1. - dis_real))
  loss += tf.reduce_mean(tf.nn.relu(1. + dis_fake))
  return loss


def loss_hinge_gen(dis_fake):
  loss = -tf.reduce_mean(dis_fake)
  return loss


def gen_feature_matching(features_real, features_fake):
  data_moments = tf.reduce_mean(features_real, axis=0)
  sample_moments = tf.reduce_mean(features_fake, axis=0)
  g_loss = tf.reduce_mean(tf.abs(data_moments - sample_moments))
  return g_loss


def disc_binary_cross_entropy(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss


def gen_binary_cross_entropy(disc_generated_output):
  positive_labels = tf.ones_like(disc_generated_output)
  gan_loss = loss_object(positive_labels, disc_generated_output)
  return gan_loss


def gen_l1_loss(generated_image, target_image, lambda_):
  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target_image - generated_image))
  return lambda_ * l1_loss


def disc_log_loss(disc_real_output, disc_generated_output, EPS=1e-12):
  disc_loss = tf.reduce_mean(-(tf.math.log(disc_real_output + EPS) + tf.math.log(1 - disc_generated_output + EPS)))
  return disc_loss


def gen_log_loss(disc_generated_output, EPS=1e-12):
  gan_loss = tf.reduce_mean(-tf.math.log(disc_generated_output + EPS))
  return gan_loss
