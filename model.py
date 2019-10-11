"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class Model(object):
  def __init__(self, config):
    assert config["model_type"] in ["cnn", "linear"]
    self.is_training = tf.placeholder(tf.bool)
    self.x_input = tf.placeholder(tf.float32, shape = [None, 28, 28, 1])
    self.y_input = tf.placeholder(tf.int64, shape = [None])

    self.transform = tf.placeholder_with_default(tf.zeros((tf.shape(self.x_input)[0], 3)), shape=[None, 3])
    trans_x, trans_y, rot = tf.unstack(self.transform, axis=1)
    rot *= np.pi / 180 # convert degrees to radians

    x = self.x_input

    #rotate and translate image
    ones = tf.ones(shape=tf.shape(trans_x))
    zeros = tf.zeros(shape=tf.shape(trans_x))
    trans = tf.stack([ones,  zeros, -trans_x,
                     zeros, ones,  -trans_y,
                     zeros, zeros], axis=1)
    x = tf.contrib.image.rotate(x, rot, interpolation='BILINEAR')
    x = tf.contrib.image.transform(x, trans, interpolation='BILINEAR')
    self.x_image = x

    ch = 1

    if config["model_type"] == "cnn":
        x.set_shape((None, 28, 28, 1))
        x = tf.layers.conv2d(x, 32, (5, 5), activation='relu', padding='same', name='conv1')
        x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='same')
        x = tf.layers.conv2d(x, 64, (5, 5), activation='relu', padding='same', name='conv2')
        x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='same')

        x = tf.layers.flatten(x)
        #x = tf.layers.flatten(tf.transpose(x, (0, 3, 1, 2)))
        x = tf.layers.dense(x, 1024, activation='relu', name='fc1')
        self.pre_softmax = tf.layers.dense(x, 10, name='fc2')
    else:
        W_fc = self._weight_variable([784*ch, 2])
        b_fc = self._bias_variable([2])
        self.W = W_fc
        self.b = b_fc
        x_flat = tf.reshape(x, [-1, 784*ch])
        self.pre_softmax = tf.matmul(x_flat, W_fc) + b_fc

    self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.y_input, logits=self.pre_softmax)

    self.xent = tf.reduce_sum(self.y_xent)
    self.mean_xent = tf.reduce_mean(self.y_xent)

    self.y_pred = tf.argmax(self.pre_softmax, 1)

    self.correct_prediction = tf.equal(self.y_pred, self.y_input)

    self.num_correct = tf.reduce_sum(tf.cast(self.correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

  @staticmethod
  def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial)

  @staticmethod
  def _conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

  @staticmethod
  def _max_pool_2x2( x):
      return tf.nn.max_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')
