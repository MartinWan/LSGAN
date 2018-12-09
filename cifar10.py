from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import pickle
import numpy as np

FLAGS = tf.app.flags.FLAGS

class CIFAR10():
  def load(self, filename):
    tmp = pickle.load(open(filename, "rb"))
    x_train = tmp['data'].astype(np.float32).reshape([-1, 32, 32, 3]) / 127.5 - 1.

    # hack to make multiple of batch_size since LSGAN nodes have hardcoded batch sizes :(
    size = FLAGS.batch_size * (x_train.shape[0] // FLAGS.batch_size)
    x_train = x_train[:size]
    x_train = tf.convert_to_tensor(x_train)

    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()

    return tf.reshape(
            iterator.get_next(),
            (FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim)
        )


